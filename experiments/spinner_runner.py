"""A runner that lives ON spinner and drains its own queue, one job at a time.

WHY THIS EXISTS. The original design has pilot run `queue_runner.py --machines spinner`, which
dispatches each job over ssh and waits. That makes every remote job a child of pilot's
wakefulness. On 2026-08-24 pilot slept minutes after two jobs started and rebooted at 09:02; both
runs died having produced nothing, and seventeen hours were lost. Spinner has been up 54 days.
The stable machine should not depend on the unstable one.

So spinner gets its own queue directory and drains it by itself. Pilot pushes jobs in and pulls
results out whenever it happens to be awake; if pilot sleeps for a week, spinner keeps working.

DESIGN, and the reasons for each choice:

  - FOUR AT A TIME, WITH ONE HEAVY. Raised from serial on 2026-08-25: run_h1_h4 is
    single-threaded (HGMISS_NJOBS defaults to 1), so a serial queue left ten of spinner's eleven
    cores idle. The limit that actually binds is MEMORY, not cores. Spinner has 18 GB and
    run_h1_h4's own comment says mask_interaction's design matrix is "tens of GB/fold", so four
    concurrent BASE-* jobs would eventually all sit in mask_interaction and exhaust the machine.
    Hence the per-job `slots` and `exclusive` fields: a job declares how much of the machine
    it needs, because the runner cannot know and guessing from the job id goes stale.
  - N_JOBS IS CAPPED for jobs that do not set it. gate_baselines defaults to N_JOBS=-1, meaning
    every core; four of those concurrently would be 44 workers on 11 cores, which is precisely
    the oversubscription that produced sixteen orphaned loky workers on 2026-08-23.
  - ATOMIC CLAIM via os.rename from pending/ to running/. Same mechanism as the original runner:
    rename either succeeds or raises, so two runners cannot take one job.
  - HEARTBEAT by touching the claimed file, so a stale claim is visible as an old mtime.
  - RESUME ON START. Anything left in running/ from a killed process is returned to pending/
    before the loop begins, because a job interrupted mid-flight has produced no output file and
    must be redone -- and `run_h1_h4` skips (cohort, method) rows that already exist, so redoing
    it is cheap.
  - NO OUTPUT SYNCING. This runner never copies a file over another. Declaring the shared
    results/main.csv as an output is what destroyed 1,850 rows on 2026-08-24; here a job writes
    where its own command says and pilot fetches it deliberately.

START IT LIKE THIS. `setsid` does not exist on macOS and spinner is a Mac, so use the repo's
own daemonize.py, which detaches properly and gives the runner ppid 1:

    ssh spinner 'cd ~/phd/papers/paper-plco-hypergraph && mkdir -p results/logs && \\
      nohup .venv/bin/python experiments/daemonize.py --log results/logs/spinner-local.log \\
      -- .venv/bin/python experiments/spinner_runner.py >/dev/null 2>&1 < /dev/null &'

Verify with `pgrep -f spinner_runner.py` and check the ppid is 1. Push work in and collect it
with `experiments/push_to_spinner.py {push,pull,status}` from pilot.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
QUEUE = ROOT / "queue-local"
DIRS = {d: QUEUE / d for d in ("pending", "running", "done", "failed")}
IDLE_SLEEP = 20          # seconds between polls
MAX_IDLE = 24 * 3600     # give up after a day with nothing to do
MAX_CONCURRENT = 4       # total slots in flight
# REDUCED FROM 8 on 2026-08-25 13:40, measured rather than guessed. Six concurrent jobs drove
# spinner's load to 54 on eleven cores while aggregate CPU sat at 49% OF ONE core's worth --
# processes blocked, not computing -- and swap to 3.1 GB used with 933 MB free. Load was still
# CLIMBING across the 15/5/1-minute averages, so it was not a startup transient. Each job loads
# its own copy of a PLCO cohort, so concurrency multiplies resident data on an 18 GB machine and
# the cost lands in swap long before it shows up as CPU.
# Do not START another job when free memory is below this. The real constraint on spinner is
# memory, not cores: 18 GB total, and run_h1_h4's own comment warns that mask_interaction's
# design matrix is "tens of GB/fold". A static slot count is a GUESS about that; this is a
# measurement, so concurrency can be raised without betting on the guess being right. Jobs
# already running are never killed for memory -- that would throw away hours -- the runner just
# stops adding to the pile until the pressure clears.
MIN_FREE_GB = 4.0
# Two limits, deliberately orthogonal, because they guard different resources:
#   MAX_CONCURRENT  slots -- CPU. How many jobs may share the machine.
#   MAX_HEAVY       count -- MEMORY. How many memory-heavy jobs may coexist.
# Slots alone cannot express this. At eight slots and two slots per BASE job, four BASE jobs
# could run, and all four would eventually be inside mask_interaction at once. MIN_FREE_GB does
# not save that case: it stops new STARTS, and these would already be running.
MAX_HEAVY = 2
# Given to any job whose command does not set N_JOBS itself. 4 x 2 = 8 workers on 11 cores.
DEFAULT_NJOBS = "2"
# Fallback for jobs written before the slots field existed. run_h1_h4 over the full method list
# includes mask_interaction, whose design matrix run_h1_h4's own comment calls "tens of GB/fold".
LEGACY_HEAVY_PREFIX = ("BASE-", "FULL-")
LEGACY_HEAVY_SLOTS = 2


def is_heavy(job: dict) -> bool:
    """Does this job want a lot of MEMORY? The job says so with `"heavy": true`; the id prefix
    is only a fallback for jobs written before the field existed."""
    if "heavy" in job:
        return bool(job["heavy"])
    return job.get("id", "").startswith(LEGACY_HEAVY_PREFIX)


HISTORY = ROOT / "RESOURCE-HISTORY.json"
DEFAULT_MEM_GB = 2.0      # for a job kind never seen before; deliberately small, since the
                          # admission check re-measures continuously and a wrong guess costs
                          # one over-admission rather than a wedged machine
MEM_HEADROOM = 1.25       # declare from history at 25% above the observed peak


def job_kind(job: dict) -> str:
    """A stable key for 'jobs like this one'. The id is unique per job and useless for
    prediction; the script plus its cohort/matrix argument is what determines the footprint."""
    cmd = job.get("cmd", "")
    m = re.search(r"experiments/(\w+)\.py", cmd)
    script = m.group(1) if m else "unknown"
    a = re.search(r"--(?:cohort|matrix)\s+(\S+)", cmd)
    return f"{script}:{a.group(1)}" if a else script


def pgid_rss_gb(pgid: int) -> float:
    """Resident memory of an entire process group, in GB.

    The group, not the process: a job is a bash parent, a python child, and however many joblib
    workers that child spawns. Measuring only the process we launched is what let four jobs
    'using 0.01 GB each' consume 15.9 GB between them on 2026-08-25.
    """
    try:
        out = subprocess.run(["ps", "-Ao", "pgid,rss"], capture_output=True, text=True,
                             timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total = 0
    for ln in out.splitlines()[1:]:
        parts = ln.split()
        if len(parts) >= 2:
            try:
                if int(parts[0]) == pgid:
                    total += int(parts[1])
            except ValueError:
                continue
    return total / (1024 ** 2)


def load_history() -> dict:
    try:
        return json.loads(HISTORY.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def record_peak(kind: str, gb: float, secs: float) -> None:
    """Append an observation. Keeps the last 10 per kind -- enough to be robust to one odd run,
    short enough that a genuine change in a script's footprint is reflected quickly."""
    h = load_history()
    e = h.setdefault(kind, {"peaks_gb": [], "secs": []})
    e["peaks_gb"] = (e["peaks_gb"] + [round(gb, 3)])[-10:]
    e["secs"] = (e["secs"] + [round(secs, 1)])[-10:]
    try:
        HISTORY.write_text(json.dumps(h, indent=2, sort_keys=True))
    except OSError:
        pass


def mem_needed(job: dict) -> float:
    """How much memory to reserve. Declared beats measured beats default.

    This replaces the `slots` and `heavy` proxies. Those were invented because there was no way
    to say what a job consumes; saying the real quantity is strictly better, and HyperQueue's
    `--resource mem=N` is the model being copied (see papers/EXPERIMENT-SCHEDULER-RESEARCH.md).
    """
    d = job.get("mem_gb")
    if isinstance(d, (int, float)) and d > 0:
        return float(d)
    peaks = load_history().get(job_kind(job), {}).get("peaks_gb") or []
    if peaks:
        return max(peaks) * MEM_HEADROOM
    return DEFAULT_MEM_GB


def free_gb() -> float:
    """Available memory in GB, on macOS or Linux.

    Linux: `MemAvailable` from /proc/meminfo, which is the kernel's own estimate of what a new
    workload can get without swapping -- the right analogue of the macOS figure below, and better
    than MemFree, which excludes reclaimable page cache.

    macOS: free + inactive pages from vm_stat. Inactive pages count because macOS reclaims them
    on demand, so treating them as used would make the runner far too timid.

    Portability matters more than it looks. This function used to call vm_stat unconditionally
    and return `inf` when the call failed -- so on Linux the memory guard did not merely degrade,
    it switched off, silently, on orca, which has 15 GB and the least headroom of any host.
    """
    if sys.platform.startswith("linux"):
        try:
            for ln in pathlib.Path("/proc/meminfo").read_text().splitlines():
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / (1024 ** 2)     # kB -> GB
        except (OSError, ValueError, IndexError):
            pass
        return float("inf")               # cannot measure -> do not block work
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return float("inf")               # cannot measure -> do not block work
    page = 4096
    for ln in out.splitlines():
        if "page size of" in ln:
            try:
                page = int(ln.split("page size of")[1].split()[0])
            except (ValueError, IndexError):
                pass
    counts = {}
    for ln in out.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            try:
                counts[k.strip()] = int(v.strip().rstrip("."))
            except ValueError:
                pass
    pages = counts.get("Pages free", 0) + counts.get("Pages inactive", 0)
    return pages * page / (1024 ** 3)


def slots_for(job: dict) -> int:
    """How many of MAX_CONCURRENT slots this job occupies. THE JOB DECIDES.

    A job declares its own appetite, because the runner cannot know it and guessing from the id
    is a heuristic that goes stale the moment someone adds a job type:

        "exclusive": true   run alone -- nothing else starts until it finishes
        "slots": 2          take two of the four slots, so at most one other heavy job
        (absent)            one slot, the ordinary case

    `exclusive` exists because the binding constraint on spinner is MEMORY, not cores: 18 GB
    total, and a single mask_interaction fold can want tens of them. A job that knows it needs
    the machine should be able to say so rather than hope the scheduler infers it.
    """
    if job.get("exclusive"):
        return MAX_CONCURRENT
    s = job.get("slots")
    if isinstance(s, int) and s > 0:
        return min(s, MAX_CONCURRENT)
    if job.get("id", "").startswith(LEGACY_HEAVY_PREFIX):
        return LEGACY_HEAVY_SLOTS
    return 1


def log(msg: str) -> None:
    """stdout only. daemonize.py captures stdout into the log file, so writing there as well
    duplicated every line -- harmless but it makes the log twice as long and half as readable,
    and a doubled line is exactly the kind of thing that later reads as two events."""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _pgid_alive(pgid: int) -> bool:
    """Is any process still in this group? `ps -g` is the effect; a recorded pid is only intent."""
    try:
        out = subprocess.run(["ps", "-g", str(pgid), "-o", "pid="],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return True                      # cannot tell -> assume alive, never duplicate a job
    return bool(out.strip())


def _find_pgid(job: dict) -> int | None:
    """The pgid of a job this runner did not start.

    Prefers the pgid the runner recorded when it claimed the job. Jobs claimed before that field
    existed carry nothing, so fall back to matching the job's own --out-dir against ps -- the
    same handle status.py uses, and unique because two jobs never share an output directory.
    """
    if job.get("pgid"):
        return int(job["pgid"])
    m = re.search(r"--out-dir\s+(\S+)", job.get("cmd", ""))
    if not m:
        return None
    try:
        out = subprocess.run(["ps", "-Ao", "pgid,args"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for ln in out.splitlines():
        if m.group(1) in ln and "bash -lc" not in ln and "ps -Ao" not in ln:
            try:
                return int(ln.split()[0])
            except (ValueError, IndexError):
                continue
    return None


def _proc_started_at(pgid: int) -> float | None:
    """Wall-clock start of a process group, from ps etime. An adopted job's t0 must come from
    the process, not from the moment of adoption -- BASE-lung was recorded as running 140.9s
    when it had been going for nine hours."""
    try:
        out = subprocess.run(["ps", "-g", str(pgid), "-o", "etime="],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for et in out.split():
        days, _, rest = et.rpartition("-")
        parts = rest.split(":")
        if not all(x.isdigit() for x in parts):
            continue
        nums = [int(x) for x in parts]
        while len(nums) < 3:
            nums.insert(0, 0)
        secs = (int(days) if days else 0) * 86400 + nums[0] * 3600 + nums[1] * 60 + nums[2]
        return time.time() - secs
    return None


class AdoptedProc:
    """Stands in for a Popen the runner does not own, so a restart can keep a job rather than
    relaunch it.

    setup() used to move every job in running/ back to pending/, reasoning that a job found
    there must belong to a runner that died. That holds for a crash and fails for a restart:
    the processes survive, the job is claimed again, and TWO processes write one output file.
    On 2026-08-25 restarting to raise MAX_CONCURRENT would have duplicated two jobs nine hours
    into a fourteen-hour stage, on a machine with no room for either duplicate.

    The exit status of a non-child is not knowable, so poll() reports 0 on disappearance.
    finish() does not rely on that alone -- it also requires the declared outputs to exist, so a
    job that died badly still lands in failed/ rather than done/.
    """

    def __init__(self, pgid: int) -> None:
        self.pid = self._pgid = pgid
        self.returncode = None

    def poll(self):
        if self.returncode is None and not _pgid_alive(self._pgid):
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        try:
            os.killpg(self._pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def setup() -> list[dict]:
    """Prepare the queue dirs and decide what to do with anything already in running/.

    Returns pool records for jobs whose processes are STILL ALIVE, so the caller adopts them
    instead of relaunching them. Only genuinely dead jobs are requeued.
    """
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    adopted: list[dict] = []
    for f in sorted(DIRS["running"].glob("*.json")):
        try:
            job = json.loads(f.read_text())
        except (OSError, ValueError):
            job = {}
        pgid = _find_pgid(job) if job else None
        if pgid is not None and _pgid_alive(pgid):
            adopted.append({
                "path": f, "job": job, "proc": AdoptedProc(pgid),
                "t0": job.get("started_at") or _proc_started_at(pgid) or time.time(),
                "slots": slots_for(job), "adopted": True,
                "id": job.get("id", f.stem), "pgid": pgid, "peak_gb": pgid_rss_gb(pgid),
                "kind": job_kind(job), "logf": None,
            })
            log(f"adopted running job {f.stem} (pgid {pgid}) -- NOT relaunching it")
            continue
        # Genuinely stranded: the runner holding it died and took its process with it.
        f.rename(DIRS["pending"] / f.name)
        log(f"requeued stranded job {f.stem}")
    return adopted


def claim(free: int, free_mem: float = 1e9) -> tuple[pathlib.Path, dict] | tuple[None, None]:
    """Take the lowest-priority-number pending job that FITS in `free` slots.

    Ties break on filename for determinism."""
    cands = []
    for p in sorted(DIRS["pending"].glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cands.append((j.get("priority", 100), p.name, p, j))
    for _, _, p, j in sorted(cands, key=lambda t: (t[0], t[1])):
        if mem_needed(j) > free_mem:
            # Not enough memory for this one. Skip rather than block: a smaller job behind it
            # can still start, and this one is admitted as soon as memory frees. An exclusive
            # job waits for the pool to drain, which is what it asked for.
            continue
        if slots_for(j) > free:
            # Not enough room. Skip it rather than block the queue -- a lighter job behind it
            # can still start, and this one is picked up as soon as a slot frees. An exclusive
            # job therefore waits for the pool to drain, which is exactly what it asked for.
            continue
        dest = DIRS["running"] / p.name
        try:
            os.rename(p, dest)
        except OSError:
            continue                      # another process won the race
        os.utime(dest, None)              # fresh mtime = heartbeat starts now
        return dest, j
    return None, None


def start(path: pathlib.Path, job: dict):
    """Launch a job without waiting for it. Returns the tracking record."""
    jid = job.get("id", path.stem)
    n = slots_for(job)
    log(f"START {jid}  slots={n}{' EXCLUSIVE' if n >= MAX_CONCURRENT else ''}: "
        f"{job.get('note', '')[:90]}")
    env = dict(os.environ)
    # Cap parallelism for anything that does not ask for a specific amount. gate_baselines
    # defaults to N_JOBS=-1, every core, and four of those at once is 44 workers on 11 cores.
    if "N_JOBS=" not in job["cmd"]:
        env["N_JOBS"] = DEFAULT_NJOBS
    # Straight to a FILE, never a pipe. Two reasons, and the second is a live bug rather than a
    # preference. (a) A pipe is only drained when the job ends, so progress is invisible until
    # it is no longer useful. (b) A pipe buffer is finite -- 64 KB on macOS -- and a job that
    # writes more than that with nobody reading BLOCKS FOREVER. The largest captured log here is
    # 18.8 KB from a PARTIAL run whose MICE folds emitted 60 convergence warnings; a full
    # four-cohort run is about four times that and crosses the limit. Writing to a file removes
    # the ceiling and makes `tail -f results/<id>.log` work while the job runs.
    logf = (ROOT / "results" / f"{jid}.log").open("w", buffering=1)
    proc = subprocess.Popen(
        ["bash", "-lc", job["cmd"]], cwd=ROOT, env=env,
        stdout=logf, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    # Persist the start time and pids into the job record. They lived only in this process's
    # memory, so nothing outside the runner could tell a job that had just started from one that
    # had been silent for hours -- status.py flagged a healthy 2h43 job as "silent >4h" because
    # the only timestamp it could see was a file left by the job's PREVIOUS run.
    try:
        rec = json.loads(path.read_text())
        rec["started_at"] = time.time()
        rec["pid"], rec["pgid"] = proc.pid, os.getpgid(proc.pid)
        path.write_text(json.dumps(rec, indent=2))
    except (OSError, ValueError, ProcessLookupError):
        pass
    return {"path": path, "job": job, "proc": proc, "t0": time.time(), "slots": n, "id": jid,
            "pgid": proc.pid, "peak_gb": 0.0, "kind": job_kind(job), "logf": logf}


def finish(rec: dict) -> None:
    job, path, proc = rec["job"], rec["path"], rec["proc"]
    jid = rec["id"]
    try:
        rec["logf"].close()
    except Exception:                                          # noqa: BLE001
        pass
    rc = proc.returncode
    secs = time.time() - rec["t0"]
    # The log is already on disk -- the job wrote it as it went. Read it back for the tail.
    try:
        out = (ROOT / "results" / f"{jid}.log").read_text(errors="ignore")
    except OSError:
        out = ""
    missing = [o for o in job.get("outputs", []) if not (ROOT / o).exists()]
    job.update(exit_code=rc, seconds=round(secs, 1), missing_outputs=missing,
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
               tail="\n".join(out.strip().splitlines()[-60:]))
    ok = rc == 0 and not missing
    (DIRS["done" if ok else "failed"] / path.name).write_text(json.dumps(job, indent=2))
    path.unlink(missing_ok=True)
    if rec.get("adopted"):
        job["peak_gb_unmeasured"] = ("adopted mid-flight; RSS sampling began after the job "
                                     "started, so its true peak is unknown")
        log(f"  not recording a peak for {jid}: adopted mid-flight, sampling is incomplete")
    else:
        record_peak(rec["kind"], rec["peak_gb"], secs)
    log(f"{'done ' if ok else 'FAIL '}{jid}  rc={rc}  {secs / 60:.1f}m  "
        f"peak {rec['peak_gb']:.2f} GB (reserved {mem_needed(job):.2f})"
        + (f"  missing={missing}" if missing else ""))


def main() -> int:
    pool: list[dict] = setup()
    log(f"spinner runner up; queue={QUEUE}; max_concurrent={MAX_CONCURRENT}"
        + (f"; adopted {len(pool)} running job(s)" if pool else ""))
    idle = 0.0
    while True:
        # Enforce each job's timeout. subprocess.run did this for free when the runner was
        # serial; Popen does not, and a hung job would otherwise hold its slots forever.
        for rec in pool:
            limit = rec["job"].get("timeout", 6 * 3600)
            if rec["proc"].poll() is None and time.time() - rec["t0"] > limit:
                log(f"TIMEOUT {rec['id']} after {limit}s -- terminating its process group")
                try:
                    os.killpg(os.getpgid(rec["proc"].pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    rec["proc"].terminate()
        # reap finished jobs, so their slots are free for this round's claims
        for rec in [r for r in pool if r["proc"].poll() is not None]:
            finish(rec)
            pool.remove(rec)
        used = sum(r["slots"] for r in pool)
        started = 0
        while used < MAX_CONCURRENT:
            if pool:                       # never block the FIRST job on memory: if nothing is
                fg = free_gb()             # running, the pressure is not ours and waiting is a
                if fg < MIN_FREE_GB:       # deadlock rather than a precaution
                    log(f"holding at {len(pool)} job(s): {fg:.1f} GB free < {MIN_FREE_GB} GB")
                    break
            # Free memory MINUS what already-running jobs may still grow into. A job that
            # has not yet reached its peak would otherwise look free to admit against.
            reserved = sum(max(0.0, mem_needed(r["job"]) - r["peak_gb"]) for r in pool)
            budget = free_gb() - MIN_FREE_GB - reserved
            path, job = claim(MAX_CONCURRENT - used, budget)
            if path is None:
                break
            rec = start(path, job)
            pool.append(rec)
            used += rec["slots"]
            started += 1
        if started:
            idle = 0.0
            log(f"in flight: {len(pool)} job(s), {used}/{MAX_CONCURRENT} slots "
                f"({', '.join(r['id'] for r in pool)})")
        if not pool:
            idle += IDLE_SLEEP
            if idle >= MAX_IDLE:
                log(f"idle for {MAX_IDLE / 3600:.0f}h with nothing to claim; exiting")
                return 0
        else:
            idle = 0.0
            for r in pool:                       # heartbeat, and sample the whole group's RSS
                try:
                    os.utime(r["path"], None)
                except OSError:
                    pass
                r["peak_gb"] = max(r["peak_gb"], pgid_rss_gb(r["pgid"]))
        time.sleep(IDLE_SLEEP)


if __name__ == "__main__":
    sys.exit(main())

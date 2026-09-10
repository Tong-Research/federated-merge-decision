"""Dispatch jobs to any machine in the homelab, and fetch finished ones back.

The successor to `push_to_spinner.py`, which hardcoded a single remote. The queue mechanism is
unchanged -- jobs are files, a job is MOVED to exactly one host so it cannot run twice, and each
host drains its own `queue-local/` -- but the host is now a parameter.

    dispatch.py status                 what every host is holding
    dispatch.py push -n 5              spread the highest-priority pending jobs across hosts
    dispatch.py push --to dugong       send them all to one host
    dispatch.py pull                   collect finished jobs and outputs from every host
    dispatch.py sync                   push code to every host without moving any job

Four things carried over from the old tool, each of which exists because of a specific incident:

  * `experiments/` and `src/` ship together. Shipping one without the other put an import error
    on spinner on 2026-08-24.
  * A declared matrix is shipped before the job is, so a job cannot land and then fail for want
    of an input.
  * `pull` REFUSES an output that is smaller on the remote than locally. A shrinking file is how
    `results/main.csv` lost 1,850 rows on 2026-08-24.
  * `pull` sweeps the directory of each declared output. A job declares the file whose absence
    means failure, not everything it writes; `BASE-lung` produced four more real result files
    that a declaration-only transfer would have left behind.

And one thing that is new. **Every remote command runs under `bash -lc`.** Dugong and orca default
to fish, where the `for d in ...; do ... done` in the old `status()` is a syntax error -- it would
have reported an empty queue on a host that was busy, which is the most dangerous wrong answer
this tool can give.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL_Q = ROOT / "queue"

# cores and RAM are from a measured probe, not a spec sheet, and are used only to refuse a job
# whose declared mem_gb cannot fit. `slower` marks a host that is fine for many parallel jobs and
# poor for one long serial one -- dugong measured 3.4x slower per core than spinner (Result 157).
# Figures are MEASURED, from homelab/docs/FLEET.md (2026-09-02), not from spec sheets or from
# `nproc`. Two of them change placement for this project's jobs:
#
#   * `cpu_all` is RSA-2048 sign/s across all threads. Dugong is the fastest machine in the fleet
#     at 19,538 and orca the slowest at 8,632 -- less than half. An earlier sha256 benchmark
#     ranked them the other way round, because it was measuring hardware SHA extensions rather
#     than the integer math these jobs actually do.
#   * `disk_mbs` is a sequential write of INCOMPRESSIBLE data with the flush inside the timing.
#     Orca is a 5400 RPM laptop HDD at 89 MB/s -- 18x slower than dugong. FLEET.md's rule is
#     blunt and correct: never put data-heavy work on orca.
#
# `rank` orders the round-robin: lower goes first among hosts that can take a job. Orca is last
# on both CPU and disk, so it earns work only when the others are full.
HOSTS: dict[str, dict] = {
    "dugong":  {"remote": "phd/papers/paper-plco-hypergraph", "cores": 8, "threads": 16,
                "ram_gb": 62, "cpu_all": 19538, "disk_mbs": 1638, "rank": 0,
                "runner": "spinner_runner.py", "arch": "x86_64-linux", "site": "home"},
    "spinner": {"remote": "phd/papers/paper-plco-hypergraph", "cores": 11, "threads": 11,
                "ram_gb": 18, "cpu_all": 16372, "disk_mbs": 3657, "rank": 1,
                "runner": "spinner_runner.py", "arch": "arm64-darwin", "site": "office"},
    "orca":    {"remote": "phd/papers/paper-plco-hypergraph", "cores": 6, "threads": 12,
                "ram_gb": 15, "cpu_all": 8632, "disk_mbs": 89, "rank": 2,
                "runner": "spinner_runner.py", "arch": "x86_64-linux", "site": "home",
                "note": "5400 RPM HDD; last resort for anything that touches disk"},
}
# Known to the fleet but not provisioned for this project. Listed so `status` can say "offline"
# rather than silently omitting a machine someone expects to see.
KNOWN_UNPROVISIONED = {"trevally": "8C/12T, 16 GB, RTX 3050 Ti; Wi-Fi only, frequently offline"}
# Leave headroom so a job never takes the last gigabyte on a host that is also someone's desktop.
RAM_HEADROOM_GB = 3.0


def rq(host: str) -> str:
    return f"{HOSTS[host]['remote']}/queue-local"


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=kw.pop("timeout", 120))


def rsh(host: str, cmd: str, timeout: int = 120):
    """Run a shell command on `host` under bash, whatever the login shell is.

    dugong and orca default to fish, where `for d in ...; do ... done` is a syntax error that
    prints to stderr and leaves stdout empty -- indistinguishable from an idle host, which is the
    most dangerous wrong answer this tool can give.

    The obvious spelling, `ssh host bash -lc <cmd>`, does NOT work: ssh joins its remaining
    arguments with spaces and hands the result to the LOGIN shell, so the login shell parses the
    command and bash never sees it intact. On spinner that produced `zsh:1: parse error near
    do`; on dugong it hung until the timeout. The command has to be quoted into a single
    argument so the login shell's only job is to invoke bash.
    """
    return sh("ssh", host, f"bash -lc {shlex.quote(cmd)}", timeout=timeout)


def _remote_size(host: str, rel: str) -> int:
    """Size in bytes, or -1. BSD stat and GNU stat take different flags; try both."""
    r = rsh(host, f"stat -f %z {HOSTS[host]['remote']}/{rel} 2>/dev/null || "
                  f"stat -c %s {HOSTS[host]['remote']}/{rel} 2>/dev/null || echo -1")
    try:
        return int(r.stdout.strip().split()[0])
    except (ValueError, IndexError):
        return -1


def sync_code(hosts: list[str]) -> None:
    for host in hosts:
        rsh(host, f"mkdir -p {HOSTS[host]['remote']}")
        ok = True
        for sub in ("experiments", "src"):
            r = sh("rsync", "-a", "--exclude", "__pycache__", "--exclude", "*.pyc",
                   f"{ROOT}/{sub}/", f"{host}:{HOSTS[host]['remote']}/{sub}/", timeout=600)
            if r.returncode:
                print(f"  {host}: rsync {sub} FAILED: {r.stderr.strip()[:140]}")
                ok = False
        if ok:
            print(f"  {host}: synced experiments/ and src/")


def _fits(job: dict, host: str) -> tuple[bool, str]:
    need = float(job.get("mem_gb", 0) or 0)
    have = HOSTS[host]["ram_gb"] - RAM_HEADROOM_GB
    if need > have:
        return False, f"needs {need:g} GB, {host} has {have:g} GB usable"
    return True, ""


def push(n: int, pattern: str | None, to: str | None) -> None:
    hosts = [to] if to else list(HOSTS)
    sync_code(hosts)
    for h in hosts:
        rsh(h, f"mkdir -p {rq(h)}/pending {rq(h)}/running {rq(h)}/done {rq(h)}/failed")

    cands = []
    for p in sorted((LOCAL_Q / "pending").glob("*.json")):
        j = json.loads(p.read_text())
        if not j.get("timeout"):
            print(f"  SKIP {j['id']}: no explicit 'timeout' (seconds); the runner's 6 h default has "
                  f"already cost 12 machine-hours -- declare one")
            continue
        if pattern and pattern not in j.get("id", ""):
            continue
        cands.append((j.get("priority", 100), p.name, p, j))
    cands.sort(key=lambda t: (t[0], t[1]))

    # Spread round-robin over the hosts that can take each job. A job naming a `machine` other
    # than "any" goes there or nowhere -- silently sending it elsewhere would break whatever
    # assumption made it name a host.
    counts = {h: 0 for h in hosts}
    moved = 0
    for _, _, p, j in cands[:n]:
        want = j.get("machine")
        if want in (None, "", "any"):
            eligible = [h for h in hosts if _fits(j, h)[0]]
        elif want in hosts:
            eligible = [want] if _fits(j, want)[0] else []
        else:
            print(f"  SKIP {j['id']}: names machine {want!r}, which is not a known host")
            continue
        if not eligible:
            why = _fits(j, want or hosts[0])[1] or "no eligible host"
            print(f"  SKIP {j['id']}: {why}")
            continue
        # fewest jobs first, then by measured rank -- so a tie goes to the faster machine
        host = min(eligible, key=lambda h: (counts[h], HOSTS[h].get("rank", 99), h))

        missing = []
        for mx in j.get("matrices", []):
            src = pathlib.Path.home() / ".cache" / "phd-matrices" / f"{mx}.npz"
            if not src.exists():
                missing.append(mx)
                continue
            have = rsh(host, f"test -f .cache/phd-matrices/{mx}.npz && echo y || echo n")
            if have.stdout.strip() == "n":
                rsh(host, "mkdir -p .cache/phd-matrices")
                r = sh("rsync", "-a", str(src), f"{host}:.cache/phd-matrices/", timeout=900)
                print(f"    shipped matrix {mx} -> {host}" if not r.returncode
                      else f"    matrix {mx} FAILED: {r.stderr.strip()[:90]}")
        if missing:
            print(f"  SKIP {j['id']}: matrices absent on pilot too: {missing}")
            continue

        r = sh("rsync", "-a", str(p), f"{host}:{rq(host)}/pending/")
        if r.returncode:
            print(f"  push FAILED for {j['id']} -> {host}: {r.stderr.strip()[:120]}")
            continue
        p.unlink()                       # MOVED, so it cannot also run here
        counts[host] += 1
        moved += 1
        print(f"  pushed {j['id']:24s} -> {host:8s} (priority {j.get('priority')})")

    left = len(list((LOCAL_Q / "pending").glob("*.json")))
    spread = ", ".join(f"{h} {c}" for h, c in counts.items() if c)
    print(f"  {moved} job(s) moved{' (' + spread + ')' if spread else ''}; {left} remain here")


def pull(hosts: list[str]) -> None:
    got = 0
    for host in hosts:
        rem = HOSTS[host]["remote"]
        for sub in ("done", "failed"):
            (LOCAL_Q / sub).mkdir(parents=True, exist_ok=True)
            listing = rsh(host, f"ls {rq(host)}/{sub}/*.json 2>/dev/null || true").stdout.split()
            for remote in listing:
                name = pathlib.Path(remote).name
                rec = rsh(host, f"cat {remote}").stdout
                try:
                    j = json.loads(rec)
                except json.JSONDecodeError:
                    continue
                for o in j.get("outputs", []):
                    local = ROOT / o
                    rsize = _remote_size(host, o)
                    if local.exists() and 0 <= rsize < local.stat().st_size:
                        print(f"  REFUSED {o} from {host}: remote {rsize}B < local "
                              f"{local.stat().st_size}B -- a shrinking output is how main.csv "
                              f"lost 1,850 rows")
                        continue
                    local.parent.mkdir(parents=True, exist_ok=True)
                    sh("rsync", "-a", f"{host}:{rem}/{o}", str(local), timeout=900)
                    print(f"  fetched {o}  ({host})")
                for o in j.get("outputs", []):
                    rdir = str(pathlib.PurePosixPath(o).parent)
                    if rdir in (".", "", "results"):
                        continue                  # never sweep results/ itself -- far too broad
                    for fname in rsh(host, f"ls {rem}/{rdir}/ 2>/dev/null || true").stdout.split():
                        rel = f"{rdir}/{fname}"
                        if rel in j.get("outputs", []):
                            continue
                        local = ROOT / rel
                        if local.exists():
                            rsize = _remote_size(host, rel)
                            if 0 <= rsize < local.stat().st_size:
                                print(f"  REFUSED sibling {rel} from {host}")
                                continue
                        local.parent.mkdir(parents=True, exist_ok=True)
                        sh("rsync", "-a", f"{host}:{rem}/{rel}", str(local), timeout=900)
                        print(f"  fetched sibling {rel}  ({host})")
                (LOCAL_Q / sub / name).write_text(rec)
                rsh(host, f"rm -f {remote}")
                print(f"  {sub}: {j.get('id')}  rc={j.get('exit_code')}  {j.get('seconds')}s"
                      f"  [{host}]")
                got += 1
        # Per host, so two hosts cannot overwrite each other's cost record.
        r = sh("rsync", "-a", f"{host}:{rem}/RESOURCE-HISTORY.json",
               str(ROOT / f"RESOURCE-HISTORY.{host}.json"), timeout=180)
        if not r.returncode:
            print(f"  fetched RESOURCE-HISTORY.{host}.json")
    print(f"  {got} finished job(s) collected from {len(hosts)} host(s)")


def status(hosts: list[str]) -> None:
    tot = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    for host in hosts:
        spec, q = HOSTS[host], rq(host)
        r = rsh(host, f"for d in pending running done failed; do "
                      f"printf '%s ' $(ls {q}/$d/*.json 2>/dev/null | wc -l); done; "
                      f"pgrep -f '[{spec['runner'][0]}]{spec['runner'][1:]}' >/dev/null "
                      f"&& printf 'ALIVE' || printf 'STOPPED'; "
                      f"printf ' '; uptime | sed 's/.*load aver[^0-9]*//' | cut -d, -f1")
        parts = r.stdout.split()
        if len(parts) < 5:
            print(f"  {host:8s} UNREACHABLE or no queue ({r.stderr.strip()[:60]})")
            continue
        p, rn, d, f, runner = parts[0], parts[1], parts[2], parts[3], parts[4]
        load = parts[5] if len(parts) > 5 else "?"
        for k, v in zip(tot, (p, rn, d, f)):
            tot[k] += int(v)
        note = f"  {spec['note']}" if spec.get("note") else ""
        print(f"  {host:8s} {spec['arch']:14s} {spec['cores']:2d}c/{spec['threads']:2d}t/"
              f"{spec['ram_gb']:2d}G cpu{spec['cpu_all']:>6d} disk{spec['disk_mbs']:>5d}  "
              f"pend {p:>3s} run {rn:>2s} done {d:>3s} fail {f:>2s}  runner {runner:<7s} "
              f"load {load}{note}")
        if len(rsh(host, f"ls {q}/running/*.json 2>/dev/null || true").stdout.split()):
            for jf in rsh(host, f"ls {q}/running/ 2>/dev/null || true").stdout.split():
                print(f"           running: {jf}")
    for h, why in KNOWN_UNPROVISIONED.items():
        print(f"  {h:8s} not provisioned for this project -- {why}")
    print(f"  {'TOTAL':8s} {'':14s} {'':34s}  pend {tot['pending']:>3d} run {tot['running']:>2d} "
          f"done {tot['done']:>3d} fail {tot['failed']:>2d}")
    print(f"  local pending: {len(list((LOCAL_Q / 'pending').glob('*.json')))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull", "status", "sync"])
    ap.add_argument("-n", type=int, default=5, help="how many jobs to push")
    ap.add_argument("--match", default=None, help="only push jobs whose id contains this")
    ap.add_argument("--to", default=None, choices=sorted(HOSTS), help="target one host")
    ap.add_argument("--hosts", default=None,
                    help="comma-separated subset for pull/status/sync (default: all)")
    a = ap.parse_args()
    hosts = [h.strip() for h in a.hosts.split(",")] if a.hosts else list(HOSTS)
    bad = [h for h in hosts if h not in HOSTS]
    if bad:
        raise SystemExit(f"  unknown host(s): {bad}; known are {sorted(HOSTS)}")
    if a.action == "push":
        push(a.n, a.match, a.to)
    elif a.action == "pull":
        pull(hosts)
    elif a.action == "sync":
        sync_code(hosts)
    else:
        status(hosts)
    return 0


if __name__ == "__main__":
    sys.exit(main())

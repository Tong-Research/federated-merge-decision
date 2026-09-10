"""Launch a long experiment so it survives the shell that started it.

Written 2026-08-12 after the PLCO run died twice mid-flight, both times when the controlling
session ended rather than from any fault in the experiment. It resumed cleanly each time --
completed (cohort, method, feature_set) units are skipped -- so nothing was lost except a
night, which is exactly the kind of loss that is invisible until someone checks.

`nohup ... & disown` is not enough here. It detaches the job from the shell's job table and
ignores SIGHUP, but the process stays in the session's process GROUP, so a teardown that
signals the group still reaches it. macOS has no `setsid(1)` binary, which is the usual fix.

So this does the standard double fork:

    fork  -> parent exits, child is orphaned and reparented to init
    setsid -> child leads a NEW session with no controlling terminal
    fork  -> the session leader exits, so the survivor can never acquire one

after which nothing signalling the original session's group can reach it. stdin comes from
/dev/null and both output streams append to the log, so the job never blocks on a terminal
that no longer exists.

Usage:
    python experiments/daemonize.py --log results/logs/x.log -- <command> [args...]

Prints the surviving PID, which is what to check later. It does NOT wait, and it deliberately
reports nothing about success: whether the work is progressing is a question for the log and
the output files, not for the launcher.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys


def daemonize(cmd, log_path, env_extra=None):
    log = pathlib.Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)

    r, w = os.pipe()                      # child reports the final PID back to us
    if os.fork() > 0:                     # ---- original process
        os.close(w)
        with os.fdopen(r) as fh:
            return int(fh.read().strip() or 0)

    os.close(r)
    os.setsid()                           # ---- new session, no controlling terminal
    if os.fork() > 0:
        os._exit(0)                       # session leader exits; survivor cannot get a tty

    with os.fdopen(w, "w") as fh:         # ---- the survivor
        fh.write(str(os.getpid()))

    fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(fd, 0)
    out = os.open(str(log), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(out, 1)
    os.dup2(out, 2)

    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    os.execvpe(cmd[0], cmd, env)          # replace this process; never returns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE, repeatable; passed to the daemonised process")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="the command, after a bare --")
    a = ap.parse_args()

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given; put it after a bare --")

    env_extra = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        env_extra[k] = v

    pid = daemonize(cmd, a.log, env_extra)
    print(f"{pid}\t{' '.join(cmd)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

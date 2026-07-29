"""Make sure nothing in the fork chain outlives the process that needs it.

The fork mode is four processes deep - kernel, supervisor, zygote, measurement
child - and every one of them holds something expensive: the supervisor a
sampler, the zygote a full copy of the prefix, the child a running cell. Killing
a link in that chain must not leave the ones below it running, which is not the
default: an orphan is reparented and keeps going, holding its memory and burning
a core with nobody left to notice.

Two kernel facilities cover it, and both have to be asked for explicitly:

* ``die_with_parent`` asks to be killed when the process above goes away. Each
  link asks for itself, so the whole chain unwinds from any break in it -
  including a Jupyter kernel that was restarted or interrupted.
* ``own_process_group`` puts a measurement in a group of its own, so a timeout
  can kill the cell *and* whatever the cell started, rather than only the
  process we happen to know the id of.

Both are Linux-specific in practice. Where they are unavailable this degrades to
the old behaviour rather than failing - a leaked process is bad, a benchmark
that will not start is worse.
"""
import os
import signal

# prctl(2)
_PR_SET_PDEATHSIG = 1


def die_with_parent(sig: int = signal.SIGKILL) -> bool:
    """Ask the kernel to send *sig* here when this process's parent goes away.

    False means the guarantee is not in place: either the kernel does not offer
    it, or - the race this exists to catch - the parent had already died by the
    time we asked, so the signal would never have come. A caller that gets False
    from the second case should leave rather than serve nobody.
    """
    parent = os.getppid()
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0) != 0:
            return False
    except Exception:
        return False
    return os.getppid() == parent


def own_process_group(pid: int = 0) -> bool:
    """Put *pid* (default: this process) at the head of its own process group.

    Called on both sides of a fork on purpose. Whichever runs first wins and the
    other one's failure is meaningless, which is the standard way to close the
    window where a child is killed before it could place itself.
    """
    try:
        os.setpgid(pid, pid)
        return True
    except OSError:
        return False


def kill_group(pid: int, sig: int = signal.SIGKILL):
    """Kill *pid* and anything it started, falling back to just *pid*."""
    try:
        os.killpg(pid, sig)
        return
    except OSError:
        pass
    try:
        os.kill(pid, sig)
    except OSError:
        pass

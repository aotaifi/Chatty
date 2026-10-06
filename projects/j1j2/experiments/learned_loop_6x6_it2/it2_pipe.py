"""Run several it2 stages sequentially in one job directory (saves queue waits).
Usage: python it2_pipe.py 'script1.py args...' 'script2.py args...' ...   ({py} = this interpreter)"""
import sys, subprocess, shlex, time
for cmd in sys.argv[1:]:
    t0 = time.time(); print(time.strftime('[%H:%M:%S]'), 'PIPE start', cmd, flush=True)
    r = subprocess.run([sys.executable] + shlex.split(cmd))
    print(time.strftime('[%H:%M:%S]'), 'PIPE end', cmd, 'rc', r.returncode, 'sec', round(time.time() - t0, 1), flush=True)
    if r.returncode != 0: sys.exit(r.returncode)

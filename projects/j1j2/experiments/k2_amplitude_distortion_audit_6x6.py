import os, math, json
import numpy as np

BASE = "/Users/aliotaifi/j1j2_vit_bench"
os.chdir(BASE)

src = open("true_k2_6x6.py").read().split("d=np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')")[0]
exec(src)

inp = np.load("/Users/aliotaifi/Chatty/projects/j1j2/results/true_k2_6x6_3471990/true_k2_6x6.npz")
xs = inp["phys_states"].astype(np.uint64)
r0_true = inp["r0phys"].astype(float)
r1_true = inp["r1phys"].astype(float)

first = []
for x in xs:
    first.extend(y for y, J, mr in neigh(int(x)))
need = np.unique(np.concatenate([xs, np.asarray(first, np.uint64)]))
ensure_r0(need)
print("R0_READY", len(need), "zcache", len(zc), "r0cache", len(r0c), flush=True)

def q1(x):
    return 1.0 if T1 - r0c[int(x)] >= 0 else -1.0

def r_projected(x):
    x = int(x)
    lx = zc[x].real
    qx = q1(x)
    r = diag(x)
    for y, J, mr in neigh(x):
        amp = math.exp(zc[int(y)].real - lx)
        r += 0.5 * J * mr * q1(y) * qx * amp
    return float(r)

r1_proj = np.asarray([r_projected(x) for x in xs])
qamp = np.abs(T1 - r0_true)
print("CORR_TRUE_LOGAMP", float(np.corrcoef(np.log(qamp), r1_true)[0,1]), flush=True)
print("CORR_PROJ_LOGAMP", float(np.corrcoef(np.log(qamp), r1_proj)[0,1]), flush=True)
print("R1_TRUE_Q", np.quantile(r1_true,[0,.01,.05,.1,.5,.9,.95,.99,1]).tolist(), flush=True)
print("R1_PROJ_Q", np.quantile(r1_proj,[0,.01,.05,.1,.5,.9,.95,.99,1]).tolist(), flush=True)

def kmeans_t(a, lo=.1, hi=.9):
    x=np.asarray(a,float)
    xc=np.clip(x,np.quantile(x,lo),np.quantile(x,hi))
    m1,m2=np.quantile(xc,[.3,.8])
    for _ in range(200):
        cut=.5*(m1+m2); left=xc<=cut
        n1=xc[left].mean(); n2=xc[~left].mean()
        if abs(n1-m1)+abs(n2-m2)<1e-12: break
        m1,m2=n1,n2
    return .5*(m1+m2)

t_true = kmeans_t(r1_true)
t_proj = kmeans_t(r1_proj)
flip_true = r1_true > t_true
flip_proj = r1_proj > t_proj
print("T_TRUE", t_true, "FLIP_TRUE", int(flip_true.sum()), float(flip_true.mean()), flush=True)
print("T_PROJ", t_proj, "FLIP_PROJ", int(flip_proj.sum()), float(flip_proj.mean()), flush=True)
print("TRUE_FLIP_K1_MARGIN", np.quantile(qamp[flip_true],[0,.25,.5,.75,1]).tolist(), flush=True)
if flip_proj.any():
    print("PROJ_FLIP_K1_MARGIN", np.quantile(qamp[flip_proj],[0,.25,.5,.75,1]).tolist(), flush=True)

out = "/Users/aliotaifi/Chatty/projects/j1j2/results/k2_amplitude_distortion_audit_6x6.npz"
np.savez_compressed(
    out,
    states=xs,
    r0=r0_true,
    r1_true=r1_true,
    r1_projected=r1_proj,
    qamp=qamp,
    t_true=t_true,
    t_projected=t_proj,
    flip_true=flip_true,
    flip_projected=flip_proj,
)
print("SAVED", out, flush=True)

def kmeans_t(a, lo=.1, hi=.9):
    x=np.asarray(a,float)
    xc=np.clip(x,np.quantile(x,lo),np.quantile(x,hi))
    m1,m2=np.quantile(xc,[.3,.8])
    for _ in range(200):
        cut=.5*(m1+m2)
        left=xc<=cut
        if not left.any() or left.all(): break
        n1=xc[left].mean(); n2=xc[~left].mean()
        if abs(n1-m1)+abs(n2-m2)<1e-12: break
        m1,m2=n1,n2
    return .5*(m1+m2)

t_true = kmeans_t(r1_true)
t_proj = kmeans_t(r1_proj)
flip_true = r1_true > t_true
flip_proj = r1_proj > t_proj
print("T_TRUE", t_true, "FLIP_TRUE", int(flip_true.sum()), float(flip_true.mean()), flush=True)
print("T_PROJ", t_proj, "FLIP_PROJ", int(flip_proj.sum()), float(flip_proj.mean()), flush=True)
print("TRUE_FLIP_K1_MARGIN", np.quantile(qamp[flip_true],[0,.25,.5,.75,1]).tolist(), flush=True)
if flip_proj.any():
    print("PROJ_FLIP_K1_MARGIN", np.quantile(qamp[flip_proj],[0,.25,.5,.75,1]).tolist(), flush=True)
np.savez_compressed("/Users/aliotaifi/Chatty/projects/j1j2/results/k2_amplitude_distortion_audit_6x6.npz",states=xs,r0=r0_true,r1_true=r1_true,r1_projected=r1_proj,qamp=qamp,t_true=t_true,t_projected=t_proj,flip_true=flip_true,flip_projected=flip_proj)

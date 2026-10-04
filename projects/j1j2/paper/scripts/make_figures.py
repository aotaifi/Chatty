from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
FIG = PAPER / "figures"
DATA = PAPER / "data"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.unicode_minus": False,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.3,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def panel(ax, letter):
    ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")

def decades(ax):
    """Log y axis labelled by the exponent only (tick k means 10^k), one tick per decade."""
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=30))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{np.log10(v):.0f}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=30))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", alpha=.25, lw=.6)

def save(fig, name):
    fig.tight_layout()
    stem = Path(name).stem
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

def chain_stats(v):
    v=np.asarray(v,float)
    cm=v.reshape(-1,64).mean(axis=0)
    return float(v.mean()), float(cm.std(ddof=1)/np.sqrt(len(cm)))

def fig4_fn_krylov_loop():
    """Exact FN/Krylov loop: plain full step, Anderson-accelerated (m=3), and the walker half step."""
    R=ROOT/"krylov_sign_structure/results"
    def load(path,ws_key,eps_key):
        h=json.load(open(path))["history"]; e0=json.load(open(path)).get("E0")
        it=np.array([x["it"] for x in h])
        if ws_key=="O_sign":
            ws=(1-np.array([x["O_sign"] for x in h]))/2
            eps=np.array([x["E_error"]/abs(e0) for x in h])
        else:
            ws=np.array([x["w_s"] for x in h]); eps=np.array([x["eps"] for x in h])
        return it,ws,eps
    runs=[(r"4 x 4, $a\leftarrow\phi_{\rm FN}$","C0","o","-",load(R/"anderson_loop/full_4x4_plain.json","w_s","eps")),
          (r"4 x 4, $a\leftarrow\phi_{\rm FN}$ + Anderson","C0","^","--",load(R/"anderson_loop/full_4x4_and_m3.json","w_s","eps")),
          (r"20-site, $a\leftarrow\phi_{\rm FN}$","C1","o","-",load(R/"anderson_loop/full_20_plain.json","w_s","eps")),
          (r"20-site, $a\leftarrow\phi_{\rm FN}$ + Anderson","C1","^","--",load(R/"anderson_loop/full_20_and_m3.json","w_s","eps"))]
    hs=R/"closed_fn_krylov_4x4_J2p5_J2zero_init_halfstep.json"
    if hs.exists():
        runs.insert(1,(r"4 x 4, $a\leftarrow\sqrt{a\,\phi_{\rm FN}}$ (walkers)","C2","s",":",load(hs,"O_sign","E_error")))
    FLOOR=1e-11
    fig,ax=plt.subplots(1,2,figsize=(7.0,2.9),gridspec_kw=dict(wspace=.38))
    for lab,c,m,ls,(it,ws,eps) in runs:
        ax[0].plot(it,np.where(ws<1e-14,FLOOR,ws),marker=m,ls=ls,ms=3,lw=.7,color=c,label=lab)
        ax[1].plot(it,np.maximum(eps,FLOOR),marker=m,ls=ls,ms=3,lw=.7,color=c,label=lab)
    ax[0].set_ylim(FLOOR/3,.1); ax[1].set_ylim(FLOOR/3,.2)
    for a_,lab,ttl,l in ((ax[0],"wrong-sign probability","sign error","a"),
                         (ax[1],"rel. energy error","energy error","b")):
        a_.set_yscale("log"); decades(a_); a_.set_xlim(-5,240)
        a_.set_xlabel("loop iteration"); a_.set_ylabel(r"$\log_{10}$ "+lab); a_.set_title(ttl); panel(a_,l)
    for a_ in ax:
        a_.set_yticks(10.0**np.array([-11,-9,-7,-5,-3,-1])); a_.set_yticklabels(["exact","-9","-7","-5","-3","-1"])
        a_.axhline(FLOOR,color="0.6",lw=.8,ls=":")
    ax[1].legend(frameon=False,fontsize=6.6,loc="upper right")
    save(fig,"fig5_fn_krylov_loop.png")

def fig_sampled_4x4():
    """III.C: FN/Krylov loop on 4x4 with phi_FN estimated from N i.i.d. samples of the exact FN mixed distribution."""
    import glob
    R=ROOT/"results/fn_krylov_loop_sampled_4x4"
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.8),gridspec_kw=dict(wspace=.4))
    cmap=plt.get_cmap("viridis")
    Ns=["1e05","1e06","1e07","1e08","1e09"]
    for k,n in enumerate(Ns):
        runs=[json.load(open(f))["history"] for f in sorted(glob.glob(str(R/f"iid_N{n}_cap_seed*.json")))]
        L=min(len(h) for h in runs)
        it=np.array([runs[0][i]["it"] for i in range(L)])
        for a_,key in ((ax[0],"w_s"),(ax[1],"eps")):
            v=np.exp(np.mean([np.log(np.maximum([h[i][key] for i in range(L)],1e-12)) for h in runs],axis=0))
            a_.plot(it,v,"o-",ms=2.6,lw=.7,color=cmap(k/(len(Ns)-.5)),label=f"N = {n.replace('e0','e')}")
    ex=json.load(open(R/"iid_exact_cap_seed0.json"))["history"]
    for a_,key in ((ax[0],"w_s"),(ax[1],"eps")):
        a_.plot([h["it"] for h in ex],[max(h[key],1e-12) for h in ex],"k--",lw=1.1,label="exact")
    for a_,lab,ttl,l in ((ax[0],"wrong-sign probability","sign error","a"),(ax[1],"rel. energy error","energy error","b")):
        a_.set_yscale("log"); decades(a_); a_.set_xlim(-1,51)
        a_.set_xlabel("loop iteration"); a_.set_ylabel(r"$\log_{10}$ "+lab); a_.set_title(ttl); panel(a_,l)
    ax[1].legend(frameon=False,fontsize=6.8,loc="center left",bbox_to_anchor=(1.02,.5),title="samples /\niteration",title_fontsize=6.8)
    save(fig,"fig6_sampled_4x4.png")

def fig_learning_4x4():
    """III.E: 4x4 loop with a network amplitude trained by SR on the frozen <H_FN> from N samples per step."""
    import glob
    R=ROOT/"results/learning_ladder_rung2"; K=ROOT/"krylov_sign_structure/results"
    FLOOR=1e-9
    fig,ax=plt.subplots(1,2,figsize=(6.8,2.9),gridspec_kw=dict(wspace=.4))
    def curve(files):
        H=[json.load(open(f))["history"] for f in files]; L=min(len(h) for h in H)
        it=np.array([H[0][i]["it"] for i in range(L)])
        g=lambda key: np.exp(np.mean([np.log(np.maximum([h[i][key] for i in range(L)],FLOOR)) for h in H],axis=0))
        return it,g("w_s"),g("eps")
    runs=[("network, N = 1e3","C0","o",sorted(glob.glob(str(R/"long_vmc_b1e3_s*.json")))),
          ("network, N = 1e4","C1","s",sorted(glob.glob(str(R/"long_vmc_b1e4_s*.json")))),
          ("network, N = 1e5","C2","^",sorted(glob.glob(str(R/"prod_vmc_b1e5_s*.json"))))]
    for lab,c,m,files in runs:
        it,ws,eps=curve(files)
        ax[0].plot(it,ws,m+"-",ms=2.6,lw=.6,color=c,label=lab); ax[1].plot(it,eps,m+"-",ms=2.6,lw=.6,color=c,label=lab)
    for lab,c,f,key in ((r"exact, $a\leftarrow\phi_{\rm FN}$","k",K/"anderson_loop/full_4x4_plain.json","w_s"),
                        (r"exact, $a\leftarrow\sqrt{a\,\phi_{\rm FN}}$","0.5",K/"closed_fn_krylov_4x4_J2p5_J2zero_init_halfstep.json","O_sign")):
        d=json.load(open(f)); h=d["history"]; it=np.array([x["it"] for x in h])
        if key=="w_s": ws=np.array([x["w_s"] for x in h]); eps=np.array([x["eps"] for x in h])
        else: ws=(1-np.array([x["O_sign"] for x in h]))/2; eps=np.array([x["E_error"] for x in h])/abs(d["E0"])
        ax[0].plot(it,np.maximum(ws,FLOOR),"--",lw=1.1,color=c,label=lab); ax[1].plot(it,np.maximum(eps,FLOOR),"--",lw=1.1,color=c,label=lab)
    for a_,lab,ttl,l in ((ax[0],"wrong-sign probability","sign error","a"),(ax[1],"rel. energy error","energy error","b")):
        a_.set_yscale("log"); decades(a_); a_.set_xlim(-3,153); a_.set_ylim(FLOOR/3,.2)
        a_.set_yticks(10.0**np.array([-9,-7,-5,-3,-1])); a_.set_yticklabels(["exact","-7","-5","-3","-1"])
        a_.set_xlabel("loop iteration"); a_.set_ylabel(r"$\log_{10}$ "+lab); a_.set_title(ttl); panel(a_,l)
    ax[1].legend(frameon=False,fontsize=6.4,loc="center left",bbox_to_anchor=(1.02,.5))
    save(fig,"fig9_learning_4x4.png")

def fig6_guides_compare():
    """III.D: one untrained iteration (ViT amplitude + Krylov signs -> FN) vs (a) ViT alone, (b) FN with the full ViT guide."""
    d=json.load(open(DATA/"fn_guides_comparison.json"))
    Ls=["6x6","8x8"]; xx=np.arange(2)
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.8),gridspec_kw=dict(wspace=.45))
    for a_,ref,title,l in ((ax[0],lambda L:d[L]["E_ViT_VMC"],"vs ViT alone","a"),
                           (ax[1],lambda L:d[L]["FN"]["vit"],"vs FN with full ViT guide","b")):
        for k,(key,lab,c,m) in enumerate((("krylov","FN, Krylov signs (ours)","C3","o"),("marshall","FN, Marshall signs","0.5","s"))):
            v=[(d[L]["FN"][key]["E_site"]-ref(L)["E_site"])*1e4 for L in Ls]
            e=[np.hypot(d[L]["FN"][key]["SE_site"],ref(L)["SE_site"])*1e4 for L in Ls]
            a_.errorbar(xx+(k-.5)*.18,v,yerr=e,fmt=m,ms=5.5,capsize=3,color=c,label=lab)
        a_.axhline(0,color="k",lw=.8)
        a_.set_xticks(xx,["6 x 6","8 x 8"]); a_.set_xlim(-.5,1.5)
        a_.set_ylabel(r"$\Delta E$ per site  [$10^{-4}$]"); a_.set_title(title); panel(a_,l)
    ax[0].legend(frameon=False,fontsize=7.2,loc="center",bbox_to_anchor=(.5,.6))
    save(fig,"fig7_guides_compare.png")

def fig6_fn_benchmarks():
    """Fixed node with Marshall or Krylov signs vs published values: (a) 6x6 with ED, (b) 8x8."""
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.4),gridspec_kw=dict(wspace=.95))

    def draw(a_,own,lit,title,exact=None):
        rows=own+[(x["method"],x["energy_per_site"],x.get("uncertainty",0.),None) for x in lit]
        rows=sorted(rows,key=lambda q:q[1],reverse=True)
        yy=np.arange(len(rows))
        for i,(name,v,e,pts) in enumerate(rows):
            col="C3" if name=="FN, Krylov signs" else ("0.45" if name.startswith("FN") or name.startswith("ViT") else "C0")
            a_.errorbar([v],[i],xerr=[e],fmt="o",ms=5,capsize=2.5,color=col)
            if pts is not None:
                a_.plot(pts,[i]*len(pts),"|",ms=10,mew=1.1,color="0.35")
        if exact is not None:
            a_.axvline(exact,color="k",ls="--",lw=.9); a_.text(exact,-.75,"ED ",fontsize=7.5,ha="right",va="center")
        a_.set_yticks(yy,[r[0] for r in rows],fontsize=7.5); a_.invert_yaxis()
        a_.set_xlabel("energy per site"); a_.set_title(title); a_.grid(axis="x",alpha=.2)
        a_.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.3f}"))

    g=json.load(open(DATA/"fn_guides_comparison.json"))
    def ours(L):
        f=g[L]["FN"]; out=[]
        for key,name in (("marshall","FN, Marshall signs"),("krylov","FN, Krylov signs")):
            out.append((name,f[key]["E_site"],f[key]["SE_site"],np.array(f[key]["rep_values"])/(36 if L=="6x6" else 64)))
        out.append(("ViT (VMC)",g[L]["E_ViT_VMC"]["E_site"],g[L]["E_ViT_VMC"]["SE_site"],None))
        out.append(("FN, ViT guide",f["vit"]["E_site"],f["vit"]["SE_site"],None))
        return out
    l6=json.load(open(DATA/"literature_6x6_pbc_J2p5.json"))
    draw(ax[0],ours("6x6"),l6["benchmarks"],"6 x 6",exact=l6["exact"])
    ax[0].set_xticks([-0.504,-0.502,-0.500])
    panel(ax[0],"a")

    lit=json.load(open(DATA/"literature_8x8_pbc_J2p5.json"))
    draw(ax[1],ours("8x8"),lit["benchmarks"],"8 x 8")
    ax[1].set_xticks([-0.498,-0.496])
    panel(ax[1],"b")
    save(fig,"fig8_benchmarks.png")

def parse_halfstep():
    lines=open(DATA/"fn_mle_halfstep_exact4x4.out").read().splitlines()
    e0=None; out={}
    for line in lines:
        if line.startswith("EXACT E0"):
            e0=float(line.split()[2])
        elif line.startswith("HALF_START"):
            sp=line.split(); lab=sp[1]
            out.setdefault(lab,{"it":[0],"E":[float(sp[3])]})
        elif line.startswith("HALF_ITER"):
            sp=line.split(); lab=sp[1]; it=int(sp[2])+1
            vals={sp[i]:sp[i+1] for i in range(3,len(sp)-1,2)}
            out.setdefault(lab,{"it":[],"E":[]})
            out[lab]["it"].append(it); out[lab]["E"].append(float(vals["halfE"]))
    return e0,out

def fig7_learning_8x8():
    d=json.load(open(DATA/"fn8_loop_iterations.json"))
    lit=json.load(open(DATA/"literature_8x8_pbc_J2p5.json"))
    rows=d["rows"]; N=d["N"]
    fig,ax=plt.subplots(figsize=(4.6,2.9))
    x=np.arange(len(rows))
    for i,r in enumerate(rows):
        e=np.array(r["E"])/N
        ax.errorbar([i],[e.mean()],yerr=[abs(e[0]-e[1])/2],fmt="o",ms=6,capsize=4,color="C0" if i==0 else "C3")
        ax.plot([i]*len(e),e,"_",ms=10,mew=1.2,color="0.4")
    best=min(b["energy_per_site"] for b in lit["benchmarks"])
    ax.axhline(best,color="0.6",ls=":",lw=1); ax.text(len(rows)-.5,best,"best literature",fontsize=7,va="bottom",ha="right",color="0.4")
    ax.set_xticks(x,["no training\n(Sec. III D)","1 learned\nupdate","same update,\nrebuilt","2 learned\nupdates"],fontsize=8)
    ax.set_ylabel("FN energy per site"); ax.set_xlim(-.5,len(rows)-.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.4f}"))
    ax.set_title("8 x 8, J2 / J1 = 0.5: learned amplitude updates")
    save(fig,"fig10_learning_8x8.png")

def fig1_proof_of_concept():
    rows=list(csv.DictReader(open(ROOT/"krylov_sign_structure/results/square_exact_energyopt.csv")))
    chosen=[]
    for r in rows:
        j=float(r["J2"])
        if (j<=.6 and r["baseline"]=="marshall") or (j>=.8 and r["baseline"]=="stripe_x"):
            chosen.append(r)
    j2=np.array([float(r["J2"]) for r in chosen])
    o0=np.array([float(r["O_S_0"]) for r in chosen])
    o1=np.array([float(r["O_S_1"]) for r in chosen])
    w0=np.maximum((1-o0)/2,1e-12)
    w1=np.maximum((1-o1)/2,1e-12)
    de0_site=np.array([float(r["fixed_amp_energy_error0_per_site"]) for r in chosen])
    de1_site=np.array([float(r["fixed_amp_energy_error1_per_site"]) for r in chosen])
    ebase=np.array([float(r["E_fixed_baseline"]) for r in chosen])
    e0=ebase-16*de0_site
    de0=np.maximum(16*de0_site/np.abs(e0),1e-12)
    de1=np.maximum(16*de1_site/np.abs(e0),1e-12)

    i05=int(np.argmin(np.abs(j2-0.5)))
    exact20=json.load(open(ROOT/"krylov_sign_structure/results/groundstate_k1_20site_exact_J2p5.json"))
    wb=np.array([w0[i05],exact20["wrong_weight_0"]])
    wa=np.array([w1[i05],exact20["wrong_weight_1"]])
    eb=np.array([de0[i05],exact20["epsilon_rel_0"]])
    ea=np.array([de1[i05],exact20["epsilon_rel_1"]])

    thr=np.load(DATA/"k1_threshold_4x4_J2p5.npz")
    r=thr["r"]; p=thr["p"]; wrong=thr["is_wrong0"]; T=float(thr["T"])

    it=json.load(open(DATA/"k1_iterated_4x4.json"))["rows"]
    FLOOR=1e-10  # exact zeros (exact ground-state signs) are drawn on this line

    fig,ax=plt.subplots(2,2,figsize=(7.2,5.6),gridspec_kw=dict(hspace=.5,wspace=.34))
    ax=ax.flat

    bins=np.linspace(-30,12,85)
    rc=np.clip(r,bins[0],bins[-1])
    ax[0].hist([rc[~wrong],rc[wrong]],bins=bins,weights=[p[~wrong],p[wrong]],
               stacked=True,label=["Marshall sign correct","Marshall sign wrong"])
    ax[0].axvline(T,color="k",ls="--",lw=1)
    ax[0].text(T+.6,.50,"threshold $T$:\nflip sign if $r>T$",transform=ax[0].get_xaxis_transform(),fontsize=7.2)
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-6,3e3); decades(ax[0]); ax[0].set_yticks(10.0**np.arange(-6,1))
    ax[0].set_xlabel(r"local energy $r(x)=(H\psi)(x)/\psi(x)$")
    ax[0].set_ylabel(r"$\log_{10}$ probability $|\psi_0(x)|^2$")
    ax[0].set_title("4 x 4, J2 / J1 = 0.5: first Krylov step")
    ax[0].legend(frameon=False,fontsize=7.2,loc="upper left",handlelength=1.2); panel(ax[0],"a")

    cmap=plt.get_cmap("viridis")
    for k,row in enumerate(it):
        st=np.array([h["step"] for h in row["history"]])
        for a_,key in ((ax[1],"wrong"),(ax[2],"eps")):
            v=np.array([h[key] for h in row["history"]]); v=np.where(v>FLOOR,v,FLOOR)
            ref="Marshall" if row["start"]=="marshall" else "stripe"
            a_.plot(st,v,"o-",ms=4.5,lw=.9,color=cmap(k/(len(it)-.5)),label=f"{row['J2']:.1f}  {ref}")
    for a_,lab,ttl,l in ((ax[1],"wrong-sign probability","4 x 4: sign error vs ED","b"),
                         (ax[2],"relative energy error","4 x 4: energy error vs ED","c")):
        a_.set_yscale("log"); a_.set_ylim(FLOOR/3,1); decades(a_)
        a_.set_yticks(10.0**np.arange(-10,1,2))
        a_.set_yticklabels(["exact"]+[f"{e}" for e in range(-8,1,2)])
        a_.axhline(FLOOR,color="0.6",lw=.8,ls=":")
        a_.set_xlabel("Krylov sign steps"); a_.set_xticks(range(0,7))
        a_.set_ylabel(r"$\log_{10}$ "+lab); a_.set_title(ttl); panel(a_,l)
    ax[1].legend(frameon=False,fontsize=7,loc="upper right",title="J2/J1  start",title_fontsize=7)

    xx=np.array([0,1,2.6,3.6]); width=.36
    before=np.r_[wb,eb]; after=np.r_[wa,ea]
    ax[3].bar(xx-width/2,before,width,label="Marshall guide")
    ax[3].bar(xx+width/2,after,width,label="one Krylov step")
    for x,b0,b1 in zip(xx,before,after):
        ax[3].text(x+width/2,b1*1.5,f"{b0/b1:.0f}x",ha="center",va="bottom",fontsize=7.5,fontweight="bold")
    ax[3].set_yscale("log"); ax[3].set_ylim(1e-4,before.max()*12); decades(ax[3])
    ax[3].set_xticks(xx,["4 x 4","20-site","4 x 4","20-site"],fontsize=8)
    ax[3].text(.5,-.2,"wrong-sign prob.",ha="center",transform=ax[3].get_xaxis_transform(),fontsize=8)
    ax[3].text(3.1,-.2,"rel. energy error",ha="center",transform=ax[3].get_xaxis_transform(),fontsize=8)
    ax[3].set_title("J2 / J1 = 0.5: two exact clusters"); ax[3].set_ylabel(r"$\log_{10}$ error")
    ax[3].legend(frameon=False,fontsize=7.2,ncol=2,loc="upper center"); panel(ax[3],"d")

    save(fig,"fig2_krylov_exact_amp.png")

def fig2_krylov_approx_amp():
    d=json.load(open(DATA/"k1_approx_amp_4x4.json"))
    FLOOR=1e-10
    fig,ax=plt.subplots(1,2,figsize=(6.4,2.75),gridspec_kw=dict(wspace=.45))
    show=[0.5,0.45,0.55,0.4,0.6]                      # amplitude source J2, best to worst
    cmap=plt.get_cmap("plasma")
    for k,j in enumerate(show):
        r=[x for x in d["rows"] if abs(x["amp_source_J2"]-j)<1e-9][0]
        st=[h["step"] for h in r["history"]]
        lab=f"{j:.2f}  ({'exact' if j==0.5 else 'F=%.2f' % r['amp_fidelity']})"
        for a_,key in ((ax[0],"wrong"),(ax[1],"eps")):
            v=np.array([h[key] for h in r["history"]]); v=np.where(v>FLOOR,v,FLOOR)
            a_.plot(st,v,"o-",ms=4.5,lw=.9,color=cmap(k/5.5),label=lab)
    for a_,lab,ttl,l in ((ax[0],"wrong-sign probability","sign error","a"),
                         (ax[1],"rel. energy error","energy error","b")):
        a_.set_yscale("log"); a_.set_ylim(FLOOR/3,1); decades(a_)
        a_.set_yticks(10.0**np.arange(-10,1,2)); a_.set_yticklabels(["exact"]+[f"{e}" for e in range(-8,1,2)])
        a_.axhline(FLOOR,color="0.6",lw=.8,ls=":")
        a_.set_xlabel("Krylov sign steps"); a_.set_xticks(range(0,9,2))
        a_.set_ylabel(r"$\log_{10}$ "+lab); a_.set_title(ttl); panel(a_,l)
    ax[0].legend(frameon=False,fontsize=6.3,title="amplitude from J2 =",title_fontsize=6.3,
                 loc="center right",bbox_to_anchor=(1.02,.42))

    save(fig,"fig3_krylov_approx_amp.png")

def fig3_krylov_vit_6x6():
    """6x6, ViT amplitude fixed: Marshall guide, 1 and 2 Krylov steps, and the ViT signs, compared with ED."""
    se=json.load(open(DATA/"ed_6x6_sign_errors_final.json"))
    ed=json.load(open(ROOT/"results/ed_6x6/ed6x6_summary.json")) if (ROOT/"results/ed_6x6/ed6x6_summary.json").exists() else None
    en=json.load(open(DATA/"vit6_energies.json"))
    E0=se["E0_per_site"]; eV=se["e_ViT"]
    k2=se["sets"]["K2"]["w"]; k1=se["sets"]["K1T"]["w"]
    w_marshall=0.019558   # exact, full ED table (ed_6x6/postprocess.json)
    labels=["Marshall\nguide","1 Krylov\nstep","2 Krylov\nsteps","ViT signs\n(ref.)"]
    w=np.array([w_marshall,k1["K1"]["w"],k2["K2"]["w"],k1["ViT"]["w"]])
    we=np.array([0,k1["K1"]["err"],k2["K2"]["err"],k1["ViT"]["err"]])
    eps=np.array([(eV+en["marshall"]["dE"]-E0)/abs(E0),(eV+en["krylov"]["dE"]-E0)/abs(E0),np.nan,(eV-E0)/abs(E0)])
    epe=np.array([en["marshall"]["SE"],en["krylov"]["SE"],np.nan,se["e_ViT_se"]])/abs(E0)
    cols=["C0","C1","C1","0.55"]
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.9),gridspec_kw=dict(wspace=.45))
    fig.suptitle(r"6 x 6, J2 / J1 = 0.5, amplitude $|\psi_{\rm ViT}|$ fixed, compared with ED",fontsize=8.5,y=1.02)
    xx=np.arange(4)
    ax[0].bar(xx,w,.6,yerr=we,capsize=3,color=cols)
    for x,v in zip(xx[1:3],w[1:3]):
        ax[0].text(x,v*1.6,f"{w[0]/v:.0f}x",ha="center",fontsize=7.5,fontweight="bold")
    ax[0].set_yscale("log"); ax[0].set_ylim(5e-5,.2); decades(ax[0])
    ax[0].set_ylabel(r"$\log_{10}$ wrong-sign probability"); ax[0].set_title("sign error"); panel(ax[0],"a")
    ok=np.isfinite(eps)
    bars=ax[1].bar(xx[ok],eps[ok],.6,yerr=epe[ok],capsize=3,color=[c for c,o in zip(cols,ok) if o])
    bars[0].set_hatch("///"); bars[0].set_facecolor("white"); bars[0].set_edgecolor("C0")
    ax[1].text(xx[2],2e-4,"n/a",ha="center",fontsize=7.5,color="0.4")
    ax[1].set_yscale("log"); ax[1].set_ylim(1e-4,.05); decades(ax[1])
    ax[1].set_ylabel(r"$\log_{10}$ rel. energy error"); ax[1].set_title("energy error"); panel(ax[1],"b")
    for a_ in ax: a_.set_xticks(xx,labels,fontsize=7.5)
    save(fig,"fig4_krylov_vit_6x6.png")

if __name__=="__main__":
    fig1_proof_of_concept()
    fig2_krylov_approx_amp()
    fig3_krylov_vit_6x6()
    fig4_fn_krylov_loop()
    fig_sampled_4x4()
    fig_learning_4x4()
    fig6_guides_compare()
    fig6_fn_benchmarks()
    fig7_learning_8x8()
    print("wrote final paper figures to",FIG)

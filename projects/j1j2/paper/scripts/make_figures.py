from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
FIG = PAPER / "figures"
DATA = PAPER / "data"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.3,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def panel(ax, letter):
    ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")

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

def fig1_method():
    fig, b = plt.subplots(figsize=(3.35,2.15))
    b.axis("off")
    boxes=[(.05,.66,.28,.20,r"guide\n$\psi_k=s_ka_k$"),
           (.60,.66,.34,.20,r"fixed node\n$\phi_k^{\rm FN}>0$"),
           (.60,.12,.34,.20,r"new amplitude\n$a_{k+1}$"),
           (.05,.12,.28,.20,r"new signs\n$s_{k+1}$")]
    for x,y,w,h,t in boxes:
        b.add_patch(Rectangle((x,y),w,h,fill=False,lw=1.2))
        b.text(x+w/2,y+h/2,t,ha="center",va="center")
    arrows=[((.33,.76),(.60,.76),"fixed signs"),
            ((.77,.66),(.77,.32),"projector\ninformation"),
            ((.60,.22),(.33,.22),"one Krylov step"),
            ((.19,.32),(.19,.66),"iterate")]
    for p0,p1,t in arrows:
        b.add_patch(FancyArrowPatch(p0,p1,arrowstyle="->",mutation_scale=11,lw=1.1))
        xm,ym=(p0[0]+p1[0])/2,(p0[1]+p1[1])/2
        b.text(xm,ym+.045,t,ha="center",va="center",fontsize=8)
    b.set_title("Separate amplitude and sign updates")
    save(fig,"fig1_method.png")

def fig2_sign_story():
    rows=list(csv.DictReader(open(ROOT/"krylov_sign_structure/results/square_exact_energyopt.csv")))
    chosen=[]
    for r in rows:
        j=float(r["J2"])
        if (j<=.6 and r["baseline"]=="marshall") or (j>=.8 and r["baseline"]=="stripe_x"):
            chosen.append(r)
    j2=np.array([float(r["J2"]) for r in chosen])
    o0=np.array([float(r["O_S_0"]) for r in chosen])
    o1=np.array([float(r["O_S_1"]) for r in chosen])
    w0=np.maximum((1-o0)/2,1e-12); w1=np.maximum((1-o1)/2,1e-12)
    de0=np.array([max(float(r["fixed_amp_energy_error0_per_site"]),1e-12) for r in chosen])
    de1=np.array([max(float(r["fixed_amp_energy_error1_per_site"]),1e-12) for r in chosen])

    z=np.load(ROOT/"results/a1_node_audit_3479622/energy_krylov_vs_vit_6x6_indep.npz")
    eM,eV,eK=chain_stats(z["eM"]),chain_stats(z["eVA"]),chain_stats(z["eVA"]+z["dKA"])
    base=eV[0]/36
    mus=np.array([eM[0]/36,eK[0]/36,eV[0]/36])
    ses=np.array([eM[1]/36,eK[1]/36,eV[1]/36])

    z8=np.load(ROOT/"results/8x8_krylov_3471544/krylov_phys8_a2_fixedT.npz")
    r=np.asarray(z8["r"],float); y=np.asarray(z8["y"],float); pred=np.asarray(z8["pred"],float)
    left=r[pred>0]; right=r[pred<0]
    t=.5*(np.max(left)+np.min(right)) if len(left) and len(right) else -28.37107876288694
    mech=json.load(open(DATA/"mechanism_6x6.json"))

    fig=plt.figure(figsize=(10.0,5.5))
    gs=fig.add_gridspec(2,6,height_ratios=[1,1.05])
    a=fig.add_subplot(gs[0,0:2]); b=fig.add_subplot(gs[0,2:4]); c=fig.add_subplot(gs[0,4:6])
    d=fig.add_subplot(gs[1,0:3]); e=fig.add_subplot(gs[1,3:6])

    a.plot(j2,w0,"o--",label="parent sign"); a.plot(j2,w1,"s-",label="one-step Krylov")
    a.set_yscale("log"); a.set_xlabel(r"$J_2/J_1$"); a.set_ylabel("wrong-sign probability mass")
    a.legend(frameon=False); panel(a,"a")

    b.plot(j2,de0,"o--",label="parent sign"); b.plot(j2,de1,"s-",label="one-step Krylov")
    b.set_yscale("log"); b.set_xlabel(r"$J_2/J_1$"); b.set_ylabel(r"fixed-amplitude $(E-E_0)/N$")
    b.legend(frameon=False); panel(b,"b")

    x=np.arange(3); rel=1e3*(mus-base); er=1e3*ses
    c.errorbar(x,rel,yerr=er,fmt="o",capsize=3); c.axhline(0,lw=.8)
    c.set_xticks(x,["Marshall","Krylov","ViT"]); c.set_ylabel(r"$10^3[E/N-(E/N)_{\rm ViT}]$")
    c.set_title(r"$6\times6$: same amplitude"); panel(c,"c")

    bins=np.linspace(np.quantile(r,.01),np.quantile(r,.99),45)
    d.hist(r[y>0],bins=bins,density=True,alpha=.55,label="ViT sign = Marshall")
    d.hist(r[y<0],bins=bins,density=True,alpha=.55,label="ViT sign $\neq$ Marshall")
    d.axvline(t,ls="--",lw=1.1,label=rf"label-free $T={t:.2f}$")
    d.set_xlabel(r"Marshall local field $r_M(x)$"); d.set_ylabel("sample density")
    d.set_title(r"$8\times8$: local-field separation"); d.legend(frameon=False,fontsize=7.8); panel(d,"d")

    vals=[mech["train"]["marshall_wrong_mass"],mech["train"]["k1_wrong_mass"],
          mech["validation"]["marshall_wrong_mass"],mech["validation"]["k1_wrong_mass"]]
    labels=["Marshall\ntrain","Krylov\ntrain","Marshall\nvalidation","Krylov\nvalidation"]
    e.bar(np.arange(4),vals); e.set_yscale("log"); e.set_xticks(np.arange(4),labels)
    e.set_ylabel("wrong-sign probability mass"); e.set_title(r"$6\times6$: physical-weight diagnostic"); panel(e,"e")
    save(fig,"fig2_sign_story.png")

def fig3_closed_loop():
    d=json.load(open(ROOT/"krylov_sign_structure/results/closed_fn_krylov_4x4_J2p5_J2zero_init_100.json"))
    h=d["history"]; e0=d["E0"]
    it=np.array([x["it"] for x in h])
    os=np.array([x["O_sign"] for x in h])
    fa=np.array([x["F_amp"] for x in h])
    eg=np.array([x["E_error"] for x in h])
    efn=np.array([np.nan if "E_FN" not in x else x["E_FN"]-e0 for x in h])
    changed=np.array([i==0 or h[i]["sign_hash"]!=h[i-1]["sign_hash"] for i in range(len(h))])

    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    signerr=np.maximum((1-os)/2,1e-12); amperr=np.maximum(1-fa,1e-12)
    ax[0].plot(it,signerr,"-",label="sign error mass")
    ax[0].plot(it,amperr,"--",label="amplitude infidelity")
    ax[0].plot(it[changed],signerr[changed],"o",ms=3,label="sign changes")
    ax[0].set_yscale("log"); ax[0].set_xlabel("feedback iteration"); ax[0].set_ylabel("error")
    ax[0].legend(frameon=False); panel(ax[0],"a")

    ax[1].plot(it,np.maximum(eg,1e-12),"-",label="guide")
    ok=np.isfinite(efn); ax[1].plot(it[ok],np.maximum(efn[ok],1e-12),"--",label="fixed node")
    ax[1].set_yscale("log"); ax[1].set_xlabel("feedback iteration"); ax[1].set_ylabel(r"$E-E_0$")
    ax[1].legend(frameon=False); panel(ax[1],"b")
    save(fig,"fig3_closed_loop.png")

def fig4_8x8_benchmark():
    lit=json.load(open(DATA/"literature_8x8_pbc_J2p5.json"))
    kz=np.load(ROOT/"results/8x8_krylov_3471544/gr_8x8_population_scaling_M128.npz")
    mz=np.load(ROOT/"results/8x8_marshall_3471545/marshall_8x8_M128_summary.npz")
    ek=np.asarray(kz["rows"],float)[:,1]/64
    em=np.asarray(mz["rows"],float)[:,1]/64
    own=[("Marshall-FN",em.mean(),abs(em[0]-em[1])/2,em),
         ("Krylov-FN",ek.mean(),abs(ek[0]-ek[1])/2,ek)]
    rows=own+[(x["method"],x["energy_per_site"],x.get("uncertainty",0.),None) for x in lit["benchmarks"]]
    rows=sorted(rows,key=lambda q:q[1],reverse=True)
    names=[x[0] for x in rows]; vals=np.array([1e3*(x[1]+.5) for x in rows]); err=np.array([1e3*x[2] for x in rows])
    yy=np.arange(len(rows))
    fig,ax=plt.subplots(figsize=(4.0,3.55))
    ax.errorbar(vals,yy,xerr=err,fmt="o",capsize=2.5)
    for i,row in enumerate(rows):
        if row[3] is not None:
            pts=1e3*(np.asarray(row[3])+.5); ax.plot(pts,[i]*len(pts),"|",ms=12,mew=1.2)
    ax.set_yticks(yy,names); ax.invert_yaxis(); ax.set_xlabel(r"$10^3(E/N+0.5)$  (lower is better)")
    ax.set_title(r"$8\times8$ PBC, $J_2/J_1=0.5$"); ax.grid(axis="x",alpha=.2)
    save(fig,"fig4_8x8_benchmark.png")

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

def fig5_amplitude_learning():
    e0,h=parse_halfstep()
    sr=json.load(open(DATA/"fnmle8_sr_lambda1_linesearch_3506041.json"))
    rows=[x for x in sr["rows"] if "eta" in x]
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    for lab,pretty,marker in [
        ("exactAbs_Marshall","exact target modulus + Marshall","o"),
        ("uniform_Marshall","uniform modulus + Marshall","s")]:
        it=np.array(h[lab]["it"]); ee=np.maximum(np.array(h[lab]["E"])-e0,1e-12)
        ax[0].plot(it,ee,marker+"-",ms=3,label=pretty)
    ax[0].set_yscale("log"); ax[0].set_xlabel("ideal MLE/FN/Krylov iteration")
    ax[0].set_ylabel(r"$E-E_0$"); ax[0].set_title(r"$4\times4$: exact amplitude benchmark")
    ax[0].legend(frameon=False,fontsize=7.5); panel(ax[0],"a")

    eta=np.array([x["eta"] for x in rows])
    tg=np.array([x["train_gain"] for x in rows]); vg=np.array([x["val_gain"] for x in rows])
    ax[1].plot(eta,tg,"o--",ms=3,label="training FN population")
    ax[1].plot(eta,vg,"s-",ms=3,label="held-out FN population")
    ax[1].axhline(0,lw=.8); ax[1].axvline(.01,ls=":",lw=1)
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"SR step size $\eta$")
    ax[1].set_ylabel("MLE log-likelihood gain"); ax[1].set_title(r"$8\times8$: cross-population SR")
    ax[1].text(.04,.09,rf"Euclidean gradient cosine $={sr['euclidean']['cos']:.3f}$",
               transform=ax[1].transAxes,fontsize=8)
    ax[1].legend(frameon=False,fontsize=7.5); panel(ax[1],"b")
    save(fig,"fig5_amplitude_learning.png")

if __name__=="__main__":
    fig2_sign_story()
    fig3_closed_loop()
    fig4_8x8_benchmark()
    fig5_amplitude_learning()
    print("wrote final paper figures to",FIG)

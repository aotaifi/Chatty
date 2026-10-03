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

def fig3_closed_loop():
    d16=json.load(open(ROOT/"krylov_sign_structure/results/closed_fn_krylov_4x4_J2p5_J2zero_init_100.json"))
    d20=json.load(open(ROOT/"krylov_sign_structure/results/closed_fn_krylov_20site_J2p5_J2zero_init.json"))

    def unpack(d):
        h=d["history"]; e0=d["E0"]
        it=np.array([x["it"] for x in h])
        os=np.array([x["O_sign"] for x in h])
        fa=np.array([x["F_amp"] for x in h])
        eg=np.array([x.get("E_relative_error",x["E_error"]/abs(e0)) for x in h])
        efn=np.array([np.nan if "E_FN" not in x else x.get("E_FN_relative_error",(x["E_FN"]-e0)/abs(e0)) for x in h])
        return it,np.maximum((1-os)/2,1e-12),np.maximum(1-fa,1e-12),np.maximum(eg,1e-12),efn

    z16=unpack(d16); z20=unpack(d20)
    fig,ax=plt.subplots(2,2,figsize=(7.2,5.4))
    for z,lab,marker in [(z16,"N = 16","o"),(z20,"N = 20","s")]:
        it,ws,amp,eg,efn=z
        ax[0,0].plot(it,ws,marker+"-",ms=2.5,label=lab)
        ax[0,1].plot(it,amp,marker+"-",ms=2.5,label=lab)
        ax[1,0].plot(it,eg,marker+"-",ms=2.5,label=lab)
        ok=np.isfinite(efn)
        ax[1,1].plot(it[ok],np.maximum(efn[ok],1e-12),marker+"-",ms=2.5,label=lab)

    ax[0,0].set_yscale("log"); ax[0,0].set_ylabel("Wrong-sign probability"); ax[0,0].set_xlabel("Feedback iteration"); ax[0,0].set_title("Sign error")
    ax[0,1].set_yscale("log"); ax[0,1].set_ylabel("1 - amplitude fidelity"); ax[0,1].set_xlabel("Feedback iteration"); ax[0,1].set_title("Amplitude error")
    ax[1,0].set_yscale("log"); ax[1,0].set_ylabel("Relative energy error"); ax[1,0].set_xlabel("Feedback iteration"); ax[1,0].set_title("Current trial state")
    ax[1,1].set_yscale("log"); ax[1,1].set_ylabel("Relative energy error"); ax[1,1].set_xlabel("Feedback iteration"); ax[1,1].set_title("Fixed-node projected state")
    for a0,l in zip(ax.flat,"abcd"):
        a0.legend(frameon=False,fontsize=7.5); panel(a0,l)
    save(fig,"fig3_closed_loop.png")

def fig4_8x8_benchmark():
    lit=json.load(open(DATA/"literature_8x8_pbc_J2p5.json"))
    kz=np.load(ROOT/"results/8x8_krylov_3471544/gr_8x8_population_scaling_M128.npz")
    mz=np.load(ROOT/"results/8x8_marshall_3471545/marshall_8x8_M128_summary.npz")
    ek=np.asarray(kz["rows"],float)[:,1]/64
    em=np.asarray(mz["rows"],float)[:,1]/64
    own=[("FN, Marshall signs",em.mean(),abs(em[0]-em[1])/2,em),
         ("FN, Krylov signs",ek.mean(),abs(ek[0]-ek[1])/2,ek)]
    rows=own+[(x["method"],x["energy_per_site"],x.get("uncertainty",0.),None) for x in lit["benchmarks"]]
    rows=sorted(rows,key=lambda q:q[1],reverse=True)
    names=[x[0] for x in rows]; vals=np.array([x[1] for x in rows]); err=np.array([x[2] for x in rows])
    yy=np.arange(len(rows))
    fig,ax=plt.subplots(figsize=(4.0,3.55))
    ax.errorbar(vals,yy,xerr=err,fmt="o",capsize=2.5)
    for i,row in enumerate(rows):
        if row[3] is not None:
            pts=np.asarray(row[3]); ax.plot(pts,[i]*len(pts),"|",ms=12,mew=1.2)
    ax.set_yticks(yy,names); ax.invert_yaxis()
    ax.set_xlabel("Energy per site E/N  (lower is better)")
    ax.set_title("Periodic 8 x 8 lattice, J2 / J1 = 0.5"); ax.grid(axis="x",alpha=.2)
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
        ("exactAbs_Marshall","exact ground-state amplitudes + Marshall signs","o"),
        ("uniform_Marshall","all amplitudes equal + Marshall signs","s")]:
        it=np.array(h[lab]["it"]); ee=np.maximum((np.array(h[lab]["E"])-e0)/abs(e0),1e-12)
        ax[0].plot(it,ee,marker+"-",ms=3,label=pretty)
    ax[0].set_yscale("log"); ax[0].set_xlabel("Feedback iteration")
    ax[0].set_ylabel("Relative energy error")
    ax[0].set_title("4 x 4: exact amplitude-learning test")
    ax[0].legend(frameon=False,fontsize=7.5); panel(ax[0],"a")

    eta=np.array([x["eta"] for x in rows])
    tg=np.array([x["train_gain"] for x in rows]); vg=np.array([x["val_gain"] for x in rows])
    ax[1].plot(eta,tg,"o--",ms=3,label="walker sample used for update")
    ax[1].plot(eta,vg,"s-",ms=3,label="independent walker sample")
    ax[1].axhline(0,lw=.8); ax[1].axvline(.01,ls=":",lw=1)
    ax[1].set_xscale("log"); ax[1].set_xlabel("SR step size")
    ax[1].set_ylabel("Change in log likelihood")
    ax[1].set_title("8 x 8: validation on independent walkers")
    ax[1].text(.04,.09,f"Gradient cosine: {sr['euclidean']['cos']:.3f}",
               transform=ax[1].transAxes,fontsize=8)
    ax[1].legend(frameon=False,fontsize=7.5); panel(ax[1],"b")
    save(fig,"fig5_amplitude_learning.png")

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

    fig=plt.figure(figsize=(7.2,5.4))
    gs=fig.add_gridspec(2,2,height_ratios=[1.05,.95],hspace=.42,wspace=.32)
    ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]

    ax[0].plot(j2,w0,"o--",label="before Krylov update")
    ax[0].plot(j2,w1,"s-",label="after Krylov update")
    ax[0].set_yscale("log"); ax[0].set_xlim(0.38,1.02)
    ax[0].set_xlabel("J2 / J1"); ax[0].set_ylabel("Wrong-sign probability")
    ax[0].set_title("4 x 4 exact sign benchmark")
    ax[0].legend(frameon=False,fontsize=7.8); panel(ax[0],"a")

    ax[1].plot(j2,de0,"o--",label="before Krylov update")
    ax[1].plot(j2,de1,"s-",label="after Krylov update")
    ax[1].set_yscale("log"); ax[1].set_xlim(0.38,1.02)
    ax[1].set_xlabel("J2 / J1"); ax[1].set_ylabel("Relative energy error")
    ax[1].set_title("4 x 4 exact energy benchmark")
    ax[1].legend(frameon=False,fontsize=7.8); panel(ax[1],"b")

    xx=np.arange(2); width=.34
    ax[2].bar(xx-width/2,wb,width,label="before")
    ax[2].bar(xx+width/2,wa,width,label="after")
    ax[2].set_yscale("log"); ax[2].set_xticks(xx,["N = 16","N = 20"])
    ax[2].set_ylabel("Wrong-sign probability"); ax[2].set_title("J2 / J1 = 0.5, exact")
    for i,(x,y) in enumerate(zip(xx,wa)):
        ax[2].text(x+width/2,y*1.7,f"{wb[i]/wa[i]:.1f}x",ha="center",va="bottom",fontsize=8,fontweight="bold")
    ax[2].legend(frameon=False,fontsize=7.8); panel(ax[2],"c")

    ax[3].bar(xx-width/2,eb,width,label="before")
    ax[3].bar(xx+width/2,ea,width,label="after")
    ax[3].set_yscale("log"); ax[3].set_xticks(xx,["N = 16","N = 20"])
    ax[3].set_ylabel("Relative energy error"); ax[3].set_title("J2 / J1 = 0.5, exact")
    for i,(x,y) in enumerate(zip(xx,ea)):
        ax[3].text(x+width/2,y*1.7,f"{eb[i]/ea[i]:.1f}x",ha="center",va="bottom",fontsize=8,fontweight="bold")
    ax[3].legend(frameon=False,fontsize=7.8); panel(ax[3],"d")

    save(fig,"fig1_proof_of_concept.png")

def fig2_larger_system_check():
    z=np.load(ROOT/"results/a1_node_audit_3479622/energy_krylov_vs_vit_6x6_indep.npz")
    mech=json.load(open(DATA/"mechanism_6x6.json"))

    before=np.array([mech["train"]["marshall_wrong_mass"],mech["validation"]["marshall_wrong_mass"]])
    after=np.array([mech["train"]["k1_wrong_mass"],mech["validation"]["k1_wrong_mass"]])

    dMV=np.asarray(z["eM"]-z["eVA"],float)
    dKV=np.asarray(z["dKA"],float)
    mMV,seMV=chain_stats(dMV); mKV,seKV=chain_stats(dKV)
    vals=np.array([mMV,mKV])/36
    errs=np.array([seMV,seKV])/36

    fig,ax=plt.subplots(1,2,figsize=(7.1,2.8))
    xx=np.arange(2); width=.34
    ax[0].bar(xx-width/2,before,width,label="before K1")
    ax[0].bar(xx+width/2,after,width,label="after K1")
    ax[0].set_yscale("log"); ax[0].set_xticks(xx,["sample A","sample B"])
    ax[0].set_ylabel("ViT-weighted sign disagreement")
    ax[0].set_title("6 x 6: sign reference = ViT")
    ax[0].legend(frameon=False,fontsize=7.8); panel(ax[0],"a")

    x=np.arange(2)
    ax[1].errorbar(x,vals,yerr=errs,fmt="o",capsize=4)
    ax[1].axhline(0,lw=.8)
    ax[1].set_xticks(x,["Marshall signs","after K1"])
    ax[1].set_ylabel("Energy/site minus ViT")
    ax[1].set_title("6 x 6: identical ViT amplitude")
    panel(ax[1],"b")
    save(fig,"fig2_larger_system_check.png")

if __name__=="__main__":
    fig1_proof_of_concept()
    fig2_larger_system_check()
    fig3_closed_loop()
    fig4_8x8_benchmark()
    fig5_amplitude_learning()
    print("wrote final paper figures to",FIG)

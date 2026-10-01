from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
FIG = PAPER / "figures"
DATA = PAPER / "data"
FIG.mkdir(exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close()

def chain_stats(v):
    v = np.asarray(v, float)
    cm = v.reshape(-1, 64).mean(axis=0)
    return float(v.mean()), float(cm.std(ddof=1) / np.sqrt(len(cm)))

def fig1_exact_k1():
    path = ROOT / "krylov_sign_structure/results/square_exact_energyopt.csv"
    rows = list(csv.DictReader(open(path)))
    chosen = []
    for r in rows:
        j = float(r["J2"])
        if (j <= 0.6 and r["baseline"] == "marshall") or (
            j >= 0.8 and r["baseline"] == "stripe_x"):
            chosen.append(r)
    j2 = np.array([float(r["J2"]) for r in chosen])
    o0 = np.array([float(r["O_S_0"]) for r in chosen])
    o1 = np.array([float(r["O_S_1"]) for r in chosen])
    e0 = np.array([float(r["fixed_amp_energy_error0_per_site"]) for r in chosen])
    e1 = np.array([float(r["fixed_amp_energy_error1_per_site"]) for r in chosen])
    w0, w1 = (1-o0)/2, (1-o1)/2

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.8))
    ax[0].plot(j2, w0, "o-", label="parent sign")
    ax[0].plot(j2, w1, "s-", label="K1")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$J_2/J_1$")
    ax[0].set_ylabel("wrong-sign probability mass")
    ax[0].legend(frameon=False)

    ax[1].plot(j2, e0, "o-", label="parent sign")
    ax[1].plot(j2, e1, "s-", label="K1")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$J_2/J_1$")
    ax[1].set_ylabel("fixed-amplitude energy error / site")
    ax[1].legend(frameon=False)
    fig.suptitle("Exact 4x4 projected-Krylov sign reconstruction")
    save("fig1_exact_k1.png")

def fig2_energy_benchmarks():
    z = np.load(ROOT / "results/a1_node_audit_3479622/energy_krylov_vs_vit_6x6_indep.npz")
    eM, eV, eK = z["eM"], z["eVA"], z["eVA"] + z["dKA"]
    vals = [chain_stats(x) for x in (eM, eK, eV)]
    mu = np.array([x[0] for x in vals]) / 36
    se = np.array([x[1] for x in vals]) / 36
    lit = json.load(open(DATA / "literature_8x8_pbc_J2p5.json"))
    methods = ["Marshall-FN", "K1-FN"] + [x["method"] for x in lit["benchmarks"]]
    k8 = np.load(ROOT / "results/8x8_krylov_3471544/gr_8x8_population_scaling_M128.npz")
    m8 = np.load(ROOT / "results/8x8_marshall_3471545/marshall_8x8_M128_summary.npz")
    ek = np.asarray(k8["rows"])[:, 1] / 64
    em = np.asarray(m8["rows"])[:, 1] / 64
    energies = [em.mean(), ek.mean()] + [x["energy_per_site"] for x in lit["benchmarks"]]
    errors = [abs(em[0]-em[1])/2, abs(ek[0]-ek[1])/2]
    errors += [x.get("uncertainty", 0.0) for x in lit["benchmarks"]]

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax[0].errorbar(["Marshall", "K1", "ViT"], mu, yerr=se, fmt="o", capsize=3)
    ax[0].set_ylabel(r"$E/N$")
    ax[0].set_title("6x6 fixed amplitude (ViT modulus)")
    ax[0].tick_params(axis="x", rotation=20)

    y = np.arange(len(methods))
    ax[1].errorbar(energies, y, xerr=errors, fmt="o", capsize=2)
    ax[1].set_yticks(y, methods)
    ax[1].invert_yaxis()
    ax[1].set_xlabel(r"$E/N$")
    ax[1].set_title(r"8x8 PBC, $J_2/J_1=0.5$")
    save("fig2_energy_benchmarks.png")

def fig3_exact_closed_loop():
    p = ROOT / "krylov_sign_structure/results/closed_fn_krylov_4x4_J2p5_J2zero_init_100.json"
    dat = json.load(open(p))
    h = dat["history"]
    it = np.array([x["it"] for x in h])
    osign = np.array([x["O_sign"] for x in h])
    famp = np.array([x["F_amp"] for x in h])
    eguide = np.array([x["E_error"] for x in h])
    efn = np.array([np.nan if "E_FN" not in x else x["E_FN"]-dat["E0"] for x in h])

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax[0].plot(it, np.maximum((1-osign)/2, 1e-12), "-o", ms=2, label="sign error mass")
    ax[0].plot(it, np.maximum(1-famp, 1e-12), "-", label="amplitude infidelity")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("FN/K1 iteration")
    ax[0].set_ylabel("error")
    ax[0].legend(frameon=False)

    ax[1].plot(it, np.maximum(eguide, 1e-12), label="guide")
    ok = np.isfinite(efn)
    ax[1].plot(it[ok], np.maximum(efn[ok], 1e-12), label="FN")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("FN/K1 iteration")
    ax[1].set_ylabel(r"$E-E_0$")
    ax[1].legend(frameon=False)
    fig.suptitle("Exact 4x4 self-correcting current-sign loop")
    save("fig3_exact_closed_loop.png")

def fig4_sr_learning():
    d0 = json.load(open(DATA / "fnmle8_sr_lambda1_linesearch_3506041.json"))
    d1 = json.load(open(DATA / "fnmle_sr_iter2_train_3506071.json"))
    r0 = [r for r in d0["rows"] if "eta" in r]
    r1 = d1["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax[0].bar(["initial", "iteration 2"], [d0["euclidean"]["cos"], d1["eu_cos"]])
    ax[0].axhline(0, lw=0.8)
    ax[0].set_ylabel("cross-replica Euclidean gradient cosine")
    ax[0].set_title("Replica gradient geometry")

    for rr, lab in [(r0, "first update"), (r1, "second update")]:
        eta = np.array([x["eta"] for x in rr])
        val = np.array([x["val_gain"] for x in rr])
        ax[1].plot(eta, val, "o-", label=lab)
    ax[1].axhline(0, lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"SR step $eta$")
    ax[1].set_ylabel("held-out log-likelihood gain")
    ax[1].legend(frameon=False)
    ax[1].set_title(r"Matrix-free SR, $lambda=1$")
    save("fig4_sr_learning.png")

def fig5_local_field():
    z = np.load(ROOT / "results/8x8_krylov_3471544/krylov_phys8_a2_fixedT.npz")
    r, y = np.asarray(z["r"]), np.asarray(z["y"])
    T = -28.37107876288694
    bins = np.linspace(np.quantile(r, .01), np.quantile(r, .99), 45)
    plt.figure(figsize=(4.5, 3.0))
    plt.hist(r[y > 0], bins=bins, density=True, alpha=.55, label="Marshall-consistent")
    plt.hist(r[y < 0], bins=bins, density=True, alpha=.65, label="non-Marshall")
    plt.axvline(T, ls="--", label=rf"$T={T:.2f}$")
    plt.xlabel(r"Marshall local field $r_M(x)$")
    plt.ylabel("density")
    plt.legend(frameon=False, fontsize=8)
    plt.title("8x8 physical-sample K1 separation")
    save("fig5_local_field.png")

if __name__ == "__main__":
    fig1_exact_k1()
    fig2_energy_benchmarks()
    fig3_exact_closed_loop()
    fig4_sr_learning()
    fig5_local_field()
    print("wrote figures to", FIG)

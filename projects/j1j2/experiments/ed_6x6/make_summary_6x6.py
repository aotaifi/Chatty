"""Combine sectors_gs.json, sectors_other.json and postprocess.json into ed6x6_summary.json."""
import json, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else "/Users/aliotaifi/Chatty-organize/projects/j1j2/results/ed_6x6"
gs = json.load(open(os.path.join(R, "sectors_gs.json")))
oth = json.load(open(os.path.join(R, "sectors_other.json")))
pp = json.load(open(os.path.join(R, "postprocess.json")))
secs = {**gs, **oth}
E0 = gs["k00_A1_p"]["energies"][0]
table = {k: {"dim": v["dim"], "E_lowest": v["energies"][0], "E_per_site": v["energies"][0] / 36,
             "excitation": v["energies"][0] - E0, "energies": v["energies"], "residuals": v["residuals"],
             "t_basis_s": v["t_basis"], "t_csr_s": v["t_csr"], "t_lanczos_s": v["t_lanczos"]}
         for k, v in sorted(secs.items(), key=lambda kv: kv[1]["energies"][0])}
gsk = min(secs, key=lambda k: secs[k]["energies"][0])
out = {
    "model": "spin-1/2 J1-J2 Heisenberg, periodic 6x6, J2/J1=0.5, S^z=0",
    "library": "lattice_symmetries 0.8.4 (conda twesterhout) basis + Hamiltonian, CSR + ARPACK eigsh",
    "ground_state_sector": gsk,
    "E0": E0, "E0_per_site": E0 / 36,
    "literature_E0_per_site": -0.50381, "literature_source": "Schulz, Ziman & Poilblanc 1996 (via Lin et al. PRB 109, 235133)",
    "gap_same_sector": gs["k00_A1_p"]["energies"][1] - E0,
    "A1p_lowest3": gs["k00_A1_p"]["energies"],
    "sectors_sorted": table,
    "postprocess": pp,
    "cluster_paths": {
        "rundir": "ws1:/home/a/A.Otaifi/chatty_ed6x6/run_gs",
        "vectors": "ws1:/home/a/A.Otaifi/chatty_ed6x6/run_gs/vectors_k00_A1_p.npy (3 lowest A1,+ eigvecs, ls basis order)",
        "states": "ws1:/home/a/A.Otaifi/chatty_ed6x6/run_gs/states_k00_A1_p.npy (ls representatives = min over 576 group images)",
    },
}
json.dump(out, open(os.path.join(R, "ed6x6_summary.json"), "w"), indent=1)
print(json.dumps({k: (v["E_lowest"], v["excitation"]) for k, v in table.items()}, indent=1))

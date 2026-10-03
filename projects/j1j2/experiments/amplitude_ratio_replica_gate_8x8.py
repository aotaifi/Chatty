import json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

L = 8
N = 64
ROOT = "/Users/aliotaifi/Chatty/projects/j1j2/results/8x8_krylov_3471544"


def spins(states):
    a = np.asarray(states, np.uint64).reshape(-1, 1)
    x = 2.0 * (((a >> np.arange(N, dtype=np.uint64)) & 1).astype(float)) - 1.0
    return x.reshape(-1, L, L)


def features(states):
    x = spins(states)
    fs = []
    names = []
    for dy in range(L):
        for dx in range(L):
            if dx == 0 and dy == 0:
                continue
            fs.append(np.mean(x * np.roll(x, shift=(dy, dx), axis=(1, 2)), axis=(1, 2)))
            names.append(f"C_{dx}_{dy}")

    p = x * np.roll(x, 1, 1) * np.roll(x, 1, 2) * np.roll(np.roll(x, 1, 1), 1, 2)
    fs.append(np.mean(p, axis=(1, 2)))
    names.append("plaquette_product")
    nn_h = x * np.roll(x, 1, 2)
    nn_v = x * np.roll(x, 1, 1)
    nnn1 = x * np.roll(np.roll(x, 1, 1), 1, 2)
    nnn2 = x * np.roll(np.roll(x, 1, 1), -1, 2)
    fs += [
        np.mean(nn_h * nn_v, axis=(1, 2)),
        np.mean(nnn1 * nnn2, axis=(1, 2)),
    ]
    names += ["nn_cross", "nnn_cross"]
    return np.stack(fs, axis=1), names


def load():
    r1 = np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10501.npz")["mixed"]
    r2 = np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10502.npz")["mixed"]
    q = np.load(f"{ROOT}/krylov_phys8_a2_fixedT.npz")["states"]
    return r1, r2, q


def balanced_xy(f, q):
    X = np.concatenate([features(f)[0], features(q)[0]])
    y = np.concatenate([np.ones(len(f)), np.zeros(len(q))])
    w = np.concatenate([
        np.full(len(f), 0.5 / len(f)),
        np.full(len(q), 0.5 / len(q)),
    ])
    return X, y, w


def sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def auc_score(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    ranks = rankdata(score, method="average")
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_eval(train_f, train_q, test_f, test_q, C=0.1):
    Xtr, ytr, wtr = balanced_xy(train_f, train_q)
    Xte, yte, _ = balanced_xy(test_f, test_q)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-10] = 1.0
    Xs = (Xtr - mu) / sd
    Xv = (Xte - mu) / sd
    Z = np.column_stack([np.ones(len(Xs)), Xs])
    Zv = np.column_stack([np.ones(len(Xv)), Xv])
    lam = 1.0 / C

    def fg(beta):
        z = Z @ beta
        p = sigmoid(z)
        eps = 1e-12
        loss = -np.sum(wtr * (ytr * np.log(p + eps) + (1-ytr) * np.log(1-p + eps)))
        loss += 0.5 * lam * np.dot(beta[1:], beta[1:])
        grad = Z.T @ (wtr * (p - ytr))
        grad[1:] += lam * beta[1:]
        return float(loss), grad

    opt = minimize(lambda b: fg(b), np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
    pv = sigmoid(Zv @ opt.x)

    auc = auc_score(yte, pv)
    wf = np.concatenate([
        np.full(len(test_f), 0.5 / len(test_f)),
        np.full(len(test_q), 0.5 / len(test_q)),
    ])
    eps = 1e-12
    ll = -np.sum(wf * (yte * np.log(pv + eps) + (1-yte) * np.log(1-pv + eps)))
    return {"beta": opt.x, "mu": mu, "sd": sd}, float(auc), float(ll)


def main():
    r1, r2, q = load()
    rng = np.random.default_rng(20260930)
    qi = rng.permutation(len(q))
    q1 = q[qi[:len(q)//2]]
    q2 = q[qi[len(q)//2:]]
    rows = []
    for name, trf, trq, tef, teq in [
        ("rep1_to_rep2", r1, q1, r2, q2),
        ("rep2_to_rep1", r2, q2, r1, q1),
    ]:
        for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            _, auc, ll = fit_eval(trf, trq, tef, teq, C=C)
            rows.append({"direction": name, "C": C, "auc": auc, "logloss": ll})
            print("RATIO_GATE", name, "C", C, "AUC", auc, "logloss", ll, flush=True)

    out = {"rows": rows}
    with open("/Users/aliotaifi/Chatty/projects/j1j2/results/amplitude_ratio_replica_gate_8x8.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

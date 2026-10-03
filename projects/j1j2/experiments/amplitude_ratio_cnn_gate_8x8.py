import json
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

jax.config.update("jax_enable_x64", False)

L = 8
N = 64
ROOT = "/Users/aliotaifi/Chatty/projects/j1j2/results/8x8_krylov_3471544"


def bits_to_spins(states):
    a = np.asarray(states, np.uint64).reshape(-1, 1)
    x = 2.0 * (((a >> np.arange(N, dtype=np.uint64)) & 1).astype(np.float32)) - 1.0
    return x.reshape(-1, L, L, 1)


def auc_score(y, score):
    y = np.asarray(y, np.int8)
    score = np.asarray(score, float)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
class ResidualCNN(nn.Module):
    width: int = 16

    @nn.compact
    def __call__(self, x):
        for _ in range(3):
            x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
            x = nn.Conv(self.width, kernel_size=(3, 3), padding="VALID")(x)
            x = nn.gelu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.width)(x)
        x = nn.gelu(x)
        return nn.Dense(1)(x)[:, 0]


def load_data():
    r1 = np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10501.npz")["mixed"]
    r2 = np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10502.npz")["mixed"]
    q = np.load(f"{ROOT}/krylov_phys8_a2_fixedT.npz")["states"]
    rng = np.random.default_rng(20260930)
    qi = rng.permutation(len(q))
    q1 = q[qi[:len(q)//2]]
    q2 = q[qi[len(q)//2:]]
    return r1, r2, q1, q2
def train_one(train_f, train_q, test_f, test_q, width=16, steps=400,
              batch=256, lr=2e-3, wd=1e-4, seed=0):
    xf = jnp.asarray(bits_to_spins(train_f))
    xq = jnp.asarray(bits_to_spins(train_q))
    vf = jnp.asarray(bits_to_spins(test_f))
    vq = jnp.asarray(bits_to_spins(test_q))
    model = ResidualCNN(width)
    params = model.init(jax.random.PRNGKey(seed), xf[:2])["params"]
    tx = optax.adamw(lr, weight_decay=wd)
    state = tx.init(params)
    rng = np.random.default_rng(seed + 12345)

    def loss_fn(p, bf, bq):
        lf = model.apply({"params": p}, bf)
        lq = model.apply({"params": p}, bq)
        return 0.5 * (jnp.mean(jax.nn.softplus(-lf)) + jnp.mean(jax.nn.softplus(lq)))

    @jax.jit
    def step(p, s, bf, bq):
        loss, grad = jax.value_and_grad(loss_fn)(p, bf, bq)
        upd, s = tx.update(grad, s, p)
        return optax.apply_updates(p, upd), s, loss

    best_auc = -1.0
    best_step = 0
    for i in range(steps):
        fi = rng.integers(0, len(xf), size=batch)
        qi = rng.integers(0, len(xq), size=batch)
        params, state, loss = step(params, state, xf[fi], xq[qi])
        if i in (0, 49, 99, 199, 299, steps - 1):
            pf = np.asarray(model.apply({"params": params}, vf))
            pq = np.asarray(model.apply({"params": params}, vq))
            y = np.concatenate([np.ones(len(pf)), np.zeros(len(pq))])
            score = np.concatenate([pf, pq])
            auc = auc_score(y, score)
            if auc > best_auc:
                best_auc = auc
                best_step = i + 1
            print("CNN_GATE_STEP", i + 1, "loss", float(loss), "val_auc", auc, flush=True)
    return {"best_step": best_step, "best_auc": float(best_auc)}


def main():
    r1, r2, q1, q2 = load_data()
    rows = []
    for name, tf, tq, vf, vq, seed in [
        ("rep1_to_rep2", r1, q1, r2, q2, 11),
        ("rep2_to_rep1", r2, q2, r1, q1, 22),
    ]:
        row = train_one(tf, tq, vf, vq, seed=seed)
        row["direction"] = name
        rows.append(row)
        print("CNN_GATE_RESULT", name, row, flush=True)
    with open("/Users/aliotaifi/Chatty/projects/j1j2/results/amplitude_ratio_cnn_gate_8x8.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()

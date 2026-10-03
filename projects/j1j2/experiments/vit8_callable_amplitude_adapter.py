import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d


class Backend:
    def __init__(self, checkpoint):
        self.L = 8
        self.N = 64
        hi = nk.hilbert.Spin(s=0.5, N=self.N)
        graph = nk.graph.Hypercube(length=self.L, n_dim=2, pbc=True, max_neighbor_order=2)
        model = ViT(
            num_layers=8, d_model=60, heads=10, L_eff=16, b=2,
            transl_invariant=True, two_dimensional=True,
        )
        apply = nkjax.HashablePartial(_logpsi_transl_2d, model.apply, 2)
        sampler = nk.sampler.MetropolisExchange(
            hi, graph=graph, d_max=2, n_chains=16, sweep_size=self.N
        )
        self.vstate = nk.vqs.MCState(
            sampler=sampler,
            apply_fun=apply,
            n_samples=16,
            variables=model.init(jax.random.PRNGKey(1234), jnp.zeros((1, self.N))),
            n_discard_per_chain=10,
        )
        with open(checkpoint, "rb") as f:
            obj = flax.serialization.msgpack_restore(f.read())
        self.vstate.variables = flax.serialization.from_state_dict(self.vstate.variables, obj)

    def _bits2x(self, states):
        a = np.asarray(states, np.uint64).reshape(-1, 1)
        bits = (a >> np.arange(self.N, dtype=np.uint64)) & 1
        return 2.0 * bits.astype(float) - 1.0

    def log_amplitude_bits(self, states):
        X = self._bits2x(states)
        out = []
        for i in range(0, len(X), 4096):
            z = self.vstate.log_value(jnp.asarray(X[i:i + 4096]))
            out.append(np.asarray(z).real)
        return np.concatenate(out)


def load(checkpoint):
    if not checkpoint:
        raise ValueError("8x8 ViT checkpoint path is required")
    return Backend(checkpoint)

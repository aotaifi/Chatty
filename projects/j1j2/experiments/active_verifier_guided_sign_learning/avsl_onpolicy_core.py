#!/usr/bin/env python3
"""Small, oracle-free AVSL measurement primitives. No training or walker launch.

The sampler MUST provide independent a^2 samples for paired_fixed_a_audit.
FN mixed walker snapshots (a*phi_FN) are NOT interchangeable with a^2.
Every candidate's final FN energy requires a separate FN projector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
import json
import math
import time

# All numbers in a ledger event are INCREMENTS, not cumulative totals.
COUNTERS = (
    "wall_s", "cpu_s", "gpu_s", "n_walker_steps", "n_neighbor_hops",
    "n_guide_model_calls", "n_guide_unique_states", "n_sign_model_calls",
    "n_sign_unique_states", "n_orbit_members_expanded", "n_feature_edges",
    "n_verifier_candidates_attempted", "n_vmc_sweeps",
    "n_fn_reprojection_steps", "n_optimizer_steps", "n_replay_examples",
    "cache_hits", "cache_misses",
)


class CostLedger:
    """Append-only JSONL record with explicit hardware/seed/chamber identity.

    Unknown GPU usage must be omitted, not guessed as zero.
    Distinct-state counts refer to the associated MeteredCallable instance,
    not unique configurations across independent episodes or restarted jobs.
    """

    def __init__(self, path: str | Path, *, run_id: str, code_commit: str,
                 chamber_id: str, arm: str, device: str, model_hash: str,
                 walker_seed: int, proposal_seed: int, training_seed: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(run_id=run_id, code_commit=code_commit,
                             chamber_id=chamber_id, arm=arm, device=device,
                             model_hash=model_hash, walker_seed=walker_seed,
                             proposal_seed=proposal_seed, training_seed=training_seed)

    def record(self, phase: str, event_type: str, **increments: int | float):
        unexpected = set(increments) - set(COUNTERS)
        if unexpected:
            raise ValueError(f"unknown cost units: {sorted(unexpected)}")
        for key, value in increments.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid increment {key}={value!r}")
        event = dict(self.metadata, phase=phase, event_type=event_type,
                     t_unix_ns=time.time_ns(), **increments)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
        return event


class MeteredCallable:
    """Local *per arm* cache; count batched callable inference plus cache misses.

    batch_fn must map a list of integer spin configurations to equally many
    callable log-amplitudes or signs. This class has no complete-basis lookup.
    """

    def __init__(self, batch_fn: Callable[[list[int]], Sequence[float]],
                 *, role: str, ledger: CostLedger, phase: str):
        if role not in ("guide", "sign"):
            raise ValueError("role must be 'guide' or 'sign'")
        self.batch_fn = batch_fn
        self.role = role
        self.ledger = ledger
        self.phase = phase
        self.cache: dict[int, float] = {}
        self.requests = 0
        self.model_calls = 0

    def many(self, states: Iterable[int]) -> list[float]:
        xs = [int(x) for x in states]
        if any(x < 0 for x in xs):
            raise ValueError("spin configuration bitstrings must be nonnegative")
        unique_missing = list(dict.fromkeys(x for x in xs if x not in self.cache))
        self.requests += len(xs)
        if unique_missing:
            w0, c0 = time.perf_counter(), time.process_time()
            values = list(self.batch_fn(unique_missing))
            elapsed, cpu = time.perf_counter() - w0, time.process_time() - c0
            if len(values) != len(unique_missing):
                raise ValueError("callable returned inconsistent batch length")
            for x, v in zip(unique_missing, values):
                if not math.isfinite(float(v)):
                    raise ValueError("non-finite callable output")
                if self.role == "sign" and float(v) not in (-1.0, 1.0):
                    raise ValueError("sign output must be ±1")
                self.cache[x] = float(v)
            self.model_calls += 1
        else:
            elapsed = cpu = 0.0
        miss = len(unique_missing)
        self.ledger.record(self.phase, f"{self.role}_callable",
                           wall_s=elapsed, cpu_s=cpu,
                           cache_hits=len(xs)-miss, cache_misses=miss,
                           **{f"n_{self.role}_model_calls": int(miss > 0),
                              f"n_{self.role}_unique_states": miss})
        return [self.cache[x] for x in xs]

    def __call__(self, state: int) -> float:
        return self.many((state,))[0]


def paired_local_difference(
    x: int,
    neighbors: Callable[[int], Iterable[tuple[int, float]]],
    log_a: Callable[[int], float],
    sign: Callable[[int], float],
    eta: Callable[[int], float],
) -> float:
    """Unbiased local summand of E[a*s] - E[a*s*eta] under x ~ a(x)^2/Z.

    neighbors enumerates each Hamiltonian matrix element H_xy (not diagonal).
    This is a *joint* sign update, not the sum of individually selected flips.
    """
    x = int(x)
    ex = float(eta(x))
    sx = float(sign(x))
    if ex not in (-1.0, 1.0) or sx not in (-1.0, 1.0):
        raise ValueError("eta and sign must be ±1")
    ax = float(log_a(x))
    out = 0.0
    for y, hxy in neighbors(x):
        y = int(y)
        ey, sy = float(eta(y)), float(sign(y))
        if ey not in (-1.0, 1.0) or sy not in (-1.0, 1.0):
            raise ValueError("eta and sign must be ±1")
        if ex != ey:
            out += 2.0 * float(hxy) * math.exp(float(log_a(y)) - ax) * sx * sy
    return out


@dataclass(frozen=True)
class PairedAudit:
    """Errors are between INDEPENDENT chains, not walker autocorrelation estimates."""
    delta_energy: float
    stderr_energy: float
    delta_energy_per_site: float
    stderr_per_site: float
    independent_chains: int
    total_samples: int


def paired_fixed_a_audit(
    chains: Sequence[Sequence[int]],
    *,
    neighbors: Callable[[int], Iterable[tuple[int, float]]],
    log_a: Callable[[int], float],
    sign: Callable[[int], float],
    eta: Callable[[int], float],
    sites: int,
    sampling_law: str,
) -> PairedAudit:
    """Report, DO NOT approve, energy gain of a previously frozen sign candidate.

    Chains must be independently seeded, equilibrated and drawn from a^2;
    the caller must diagnose equilibration/autocorrelation. This function does
    not certify a sampler or account for adaptive multiple comparisons.
    """
    if sampling_law != "a2":
        raise ValueError("paired fixed-a estimator requires a^2 sampling, not FN mixed")
    if sites < 1 or len(chains) < 2 or any(len(c) == 0 for c in chains):
        raise ValueError("need positive N and at least two nonempty independent chains")
    means = [
        sum(paired_local_difference(x, neighbors, log_a, sign, eta) for x in c) / len(c)
        for c in chains
    ]
    # Equal chain means assumes equal effective sample size. Report chain-level
    # dispersion rather than treating correlated walker snapshots as iid.
    n = len(means)
    m = sum(means) / n
    se = math.sqrt(sum((v-m)**2 for v in means) / (n*(n-1)))
    return PairedAudit(m, se, m/sites, se/sites, n, sum(map(len, chains)))


def marked_window_component_parity(marked_bond_sets: Iterable[Iterable[int]]) -> tuple[int, int]:
    """Connected-component parity of marked windows sharing physical bond IDs.

    Only the combiner, not threshold selection or site-to-window construction.
    For a translation-covariant component rule, the caller must generate all
    translated windows and apply a translation-invariant marking predicate.
    """
    sets = [set(bonds) for bonds in marked_bond_sets]
    if any(not s for s in sets):
        raise ValueError("marked windows require nonempty physical bond sets")
    parent = list(range(len(sets)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sets)):
        for j in range(i):
            if sets[i] & sets[j]:
                parent[find(i)] = find(j)
    n = len({find(i) for i in range(len(sets))})
    return (-1 if n % 2 else 1), n

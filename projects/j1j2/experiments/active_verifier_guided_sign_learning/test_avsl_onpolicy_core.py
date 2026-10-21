#!/usr/bin/env python3
"""Small deterministic invariants; DOES NOT validate walker sampling or FN gains."""
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

from avsl_onpolicy_core import (
    CostLedger, MeteredCallable, paired_local_difference,
    paired_fixed_a_audit, marked_window_component_parity,
)

BASIS = [x for x in range(16) if x.bit_count() == 2]
BONDS = ((0, 1), (1, 2), (2, 3), (3, 0))


def neighbors(x):
    for i, j in BONDS:
        if ((x >> i) ^ (x >> j)) & 1:
            yield x ^ (1 << i) ^ (1 << j), 0.5


def s(x):
    return -1 if x & 1 else 1


def eta_for(chosen):
    return lambda x: -1 if x in chosen else 1


def rayleigh(log_a, sign):
    norm = sum(math.exp(2*log_a(x)) for x in BASIS)
    numerator = sum(
        math.exp(log_a(x) + log_a(y)) * sign(x) * sign(y) * h
        for x in BASIS for y, h in neighbors(x)
    )
    return numerator / norm


def exact_delta(log_a, chosen):
    eta = eta_for(chosen)
    return sum(
        math.exp(2*log_a(x)) * paired_local_difference(x, neighbors, log_a, s, eta)
        for x in BASIS
    ) / sum(math.exp(2*log_a(x)) for x in BASIS)


class FixedAmplitudeIdentityTests(unittest.TestCase):
    def test_single_and_joint_energy_match_direct_full_h(self):
        log_a = lambda x: math.log(1.0 + 0.07*((7*x) % 9))
        for chosen in (set(), {3}, {5}, {3, 5}, {3, 5, 6}, set(BASIS)):
            computed = exact_delta(log_a, chosen)
            direct = rayleigh(log_a, s) - rayleigh(
                log_a, lambda x: s(x) * eta_for(chosen)(x)
            )
            self.assertAlmostEqual(computed, direct, places=13)

    def test_joint_updates_are_not_sum_of_single_changes(self):
        log_a = lambda x: 0.0
        singles = exact_delta(log_a, {3}) + exact_delta(log_a, {5})
        jointly = exact_delta(log_a, {3, 5})
        self.assertGreater(abs(singles - jointly), 1e-5)

    def test_global_flip_and_keep_exactly_zero(self):
        log_a = lambda x: math.log(1 + 0.01*x)
        self.assertAlmostEqual(exact_delta(log_a, set()), 0)
        self.assertAlmostEqual(exact_delta(log_a, set(BASIS)), 0)

    def test_sampler_guard_rejects_fn_mixed_population(self):
        with self.assertRaisesRegex(ValueError, "a\\^2"):
            paired_fixed_a_audit(
                [BASIS, BASIS], neighbors=neighbors, log_a=lambda x: 0.0,
                sign=s, eta=eta_for({3}), sites=4, sampling_law="FN_mixed"
            )

    def test_uniform_a2_full_basis_audit(self):
        # Synthetic exact a^2 distribution: a = 1, q is uniform.
        chains = [BASIS[:], BASIS[::-1], BASIS[:]]
        a = paired_fixed_a_audit(
            chains, neighbors=neighbors, log_a=lambda x: 0.0, sign=s,
            eta=eta_for({3, 5}), sites=4, sampling_law="a2"
        )
        self.assertEqual(a.total_samples, 18)
        self.assertAlmostEqual(a.delta_energy, exact_delta(lambda x: 0.0, {3, 5}), places=14)
        self.assertAlmostEqual(a.stderr_energy, 0.0, places=14)

    def test_component_combiner_merges_overlapping_windows(self):
        self.assertEqual(marked_window_component_parity([]), (1, 0))
        self.assertEqual(marked_window_component_parity([{1, 2}, {2, 3}]), (-1, 1))
        self.assertEqual(marked_window_component_parity([{1}, {3}]), (1, 2))
        self.assertEqual(marked_window_component_parity([{2, 3}, {1, 2}]), (-1, 1))
        self.assertEqual(marked_window_component_parity([{1, 2}, {2, 3}, {8}]), (1, 2))


class CostTests(unittest.TestCase):
    def test_independent_cache_and_append_only_counters(self):
        with tempfile.TemporaryDirectory() as td:
            meta = dict(run_id="unit", code_commit="deadbeef", model_hash="sha",
                        chamber_id="post", arm="geometry", device="cpu",
                        walker_seed=11, proposal_seed=12, training_seed=13)
            file = Path(td)/"events.jsonl"
            ledger = CostLedger(file, **meta)
            fn = lambda xs: [math.log(x+1) for x in xs]
            c = MeteredCallable(fn, role="guide", ledger=ledger, phase="frontier")
            self.assertEqual(len(c.many([1, 2, 1])), 3)
            c.many([2, 3])
            self.assertEqual(c.model_calls, 2)
            self.assertEqual(len(c.cache), 3)
            events = [json.loads(line) for line in file.read_text().splitlines()]
            self.assertEqual(sum(e["n_guide_unique_states"] for e in events), 3)
            self.assertEqual(sum(e["cache_hits"] for e in events), 2)
            self.assertEqual(sum(e["n_guide_model_calls"] for e in events), 2)
            self.assertEqual(sum(e["cache_misses"] for e in events), 3)
            self.assertTrue(all(e["code_commit"] == "deadbeef" for e in events))
            sign_fn = MeteredCallable(lambda xs: [1 for x in xs], role="sign",
                                      ledger=ledger, phase="frontier")
            sign_fn.many([1, 1])
            all_events = [json.loads(line) for line in file.read_text().splitlines()]
            self.assertEqual(all_events[-1]["n_sign_unique_states"], 1)
            self.assertEqual(all_events[-1]["n_sign_model_calls"], 1)
            with self.assertRaises(ValueError):
                ledger.record("test", "bad", n_optimizer_steps=-1)
            with self.assertRaises(ValueError):
                ledger.record("test", "bad", unsupported_counter=1)

    def test_missing_values_not_accounted_as_free_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = CostLedger(Path(td)/"log.jsonl", run_id="t", code_commit="c",
                                chamber_id="ch", arm="a", device="cpu", model_hash="none",
                                walker_seed=1, proposal_seed=1, training_seed=1)
            event = ledger.record("scout", "unit", n_verifier_candidates_attempted=1)
            self.assertNotIn("gpu_s", event)


if __name__ == "__main__":
    unittest.main(verbosity=2)

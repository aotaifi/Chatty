"""pp2_lemma_bench_gate.py RUN_DIR: prints the verdict line of a pp2_lemma_bench.py run (bench.json) - timing summary, no pass/fail."""
import sys, json
R = json.load(open(sys.argv[1] + '/bench.json')); r = R.get('runs', {})
s = '; '.join(f"{k} aug {v['s_per_walker_step_aug']:.3g} s/ws (lemma {v['s_per_walker_step_lemma']:.2g}, cnn {v['s_per_walker_step_cnn']:.2g}) J/P {v['J_over_P']:.0f} peak {v['peak_GiB']:.1f}G"
              + (f" f32maxrel r2 {v['f32_vs_f64']['r2_maxrel']:.1e}" if 'f32_vs_f64' in v else '') for k, v in r.items())
so = R.get('sign_options_f64', {})
print(f"VERDICT pp2_lemma_bench {'DONE' if 'p1_ref' in R else 'INCOMPLETE'}: {s}; signB/A {so.get('B_over_A', float('nan')):.2f}")

import json, glob, sys
O = sys.argv[1]; out = None
for fn in sorted(glob.glob(O + '/a40/s[1-4].json')):
    t = json.load(open(fn))
    if out is None:
        out = {k: v for k, v in t.items() if k != 'results'}; out['results'] = {}; out['stage_files'] = []
    out['results'].update(t['results']); out['stage_files'].append(fn)
json.dump(out, open(O + '/time_a40.json', 'w'), indent=1)
print('merged', len(out['results']), 'entries from', out['stage_files'])

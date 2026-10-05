"""Validate published evidence without importing Isaac Sim or loading pickle files."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    m=json.loads((ROOT/'manifest.json').read_text())
    assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256']
    for name,digest in m['distribution_files'].items():
        assert sha(ROOT/'evaluation/team_v21'/name)==digest,name
    for e in json.loads((ROOT/'overlay_manifest.json').read_text())['files']:
        assert sha(ROOT/'overlay'/e['path'])==e['sha256'],e['path']
    for e in json.loads((ROOT/'artifacts/checkpoints/final_models_manifest.json').read_text()):
        assert sha(ROOT/e['path'])==e['sha256'],e['name']
    for e in json.loads((ROOT/'artifacts/media/manifest.json').read_text()):
        assert sha(ROOT/e['path'])==e['sha256'],e['path']
    for p in (ROOT/'overlay').rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    results=[]
    for label in ('SeenControl','UnseenEasy','UnseenMedium','UnseenHard'):
        a=json.loads((ROOT/f'results/team_v21/final/{label}.json').read_text())
        b=json.loads((ROOT/f'results/team_v21/video_repeat/{label}.json').read_text())
        assert a['summary']==b['summary'] and a['episodes']==b['episodes'],f'{label}: repeat differs'
        for d in (a,b):
            assert d['checkpoint_sha256']==m['checkpoint_sha256']
            assert d['completed']==100 and d['seed']==24 and d['max_episode_steps']==960
            assert d['observation_dimension']==123 and d['reward_protocol']=='original_Ant_7_terms_v1'
            ep=d['episodes'];assert len(ep)==100 and {e['env_id'] for e in ep}==set(range(100))
            for field in ('reward','steps','forward_displacement_m'):
                values=[e[field] for e in ep]
                assert math.isclose(statistics.mean(values),d['summary'][field]['mean'],abs_tol=1e-8)
                assert math.isclose(statistics.pstdev(values),d['summary'][field]['std'],abs_tol=1e-8)
            rate=sum(e['steps']==960 and e['timed_out'] and not e['terminated'] for e in ep)/100
            assert math.isclose(rate,d['summary']['timeout_without_failure_rate'])
        results.append(a)
    rows=list(csv.DictReader((ROOT/'results/team_v21/final/result_template.csv').open()))
    assert len(rows)==4
    for r,d in zip(rows,results):
        assert r['task_id']==d['task'] and r['checkpoint_sha256']==m['checkpoint_sha256']
        assert math.isclose(float(r['reward_mean']),d['summary']['reward']['mean'])
    docs=[ROOT/'README.md']+[ROOT/'docs'/n for n in ('METHOD.md','IMPLEMENTATION.md','RESULTS.md','REPRODUCE.md','MEDIA.md','PUBLICATION.md')]
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https://','http://','#')):continue
            assert (p.parent/target.split('#')[0]).exists(),f'{p.name}: broken link {target}'
    print('PASS: checkpoint/distribution/overlay/media hashes; 800 recorded episode statistics; video repeat equality; Python syntax; documentation links')

if __name__=='__main__':verify()

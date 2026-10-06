"""Validate published evidence without importing Isaac Sim or loading pickle files."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import tarfile
import zipfile

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
    for e in json.loads((ROOT/'artifacts/checkpoints/manifest.json').read_text()):
        assert sha(ROOT/e['checkpoint'])==e['sha256'],e['model']
    checkpoints=list((ROOT/'artifacts/checkpoints').rglob('*.pt'))
    assert len(checkpoints)==9
    assert all(not re.search(r'\d',str(p.relative_to(ROOT/'artifacts/checkpoints'))) for p in checkpoints)
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
    bundle=json.loads((ROOT/'artifacts/project/manifest.json').read_text())
    archive=ROOT/'artifacts/project'/bundle['archive']
    assert sha(archive)==bundle['sha256']
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        assert len(members)==bundle['file_count']
        for entry in members:
            rel=Path(entry.name).relative_to('IsaacLab_RS_final')
            assert not rel.is_absolute() and '..' not in rel.parts and entry.isfile()
            assert hashlib.sha256(tar.extractfile(entry).read()).hexdigest()==bundle['files'][str(rel)]
    ant='source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py'
    assert sha(ROOT/'overlay'/ant)==sha(ROOT/'reference/original/ant_env_cfg.py')
    assert not list((ROOT/'overlay').rglob('*.usd'))
    with zipfile.ZipFile(ROOT/'submission/Ant_Unseen_Terrain_5min.pptx') as z:
        assert len([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)])==7
        assert len([n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)])==7
    command=(ROOT/'submission/evaluation_command.txt').read_text()
    assert len(re.findall(r'^\s*\./isaaclab\.sh -p ', command, re.MULTILINE))==1 and '--seed 24' in command and '--num_envs 100' in command
    assert '--task "<평가 환경>"' in command
    assert '--checkpoint artifacts/checkpoints/final/ant_final.pt' in command
    assert 'python scripts/prepare_project.py' in command and './isaaclab.sh -i rsl_rl' in command
    assert 'git clone https://github.com/Kim-KiSuk/Robotics-Simulation-Assignment-1.git' in command
    docs=[ROOT/'README.md',ROOT/'artifacts/checkpoints/README.md']+[ROOT/'docs'/n for n in ('METHOD.md','IMPLEMENTATION.md','RESULTS.md','REPRODUCE.md','MEDIA.md','PUBLICATION.md','SELF_EVALUATION.md','ASSIGNMENT_CHECK.md')]
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https://','http://','#')):continue
            assert (p.parent/target.split('#')[0]).exists(),f'{p.name}: broken link {target}'
    print('PASS: checkpoint/distribution/overlay/media hashes; 800 recorded episode statistics; video repeat equality; Python syntax; documentation links')
    print('PASS: project bundle inventory/hashes; unchanged original robot configuration; no USD overlay; seven slides/notes; one submission evaluation command')

if __name__=='__main__':verify()

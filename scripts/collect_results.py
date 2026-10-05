"""Fill the official template from completed inference results; no simulation."""
import argparse
import csv
import json
from pathlib import Path
import math
import yaml

ROOT=Path(__file__).resolve().parents[1]
LEVELS={'SeenControl':('seen_control',51004),'UnseenEasy':('unseen_easy',53001),
        'UnseenMedium':('unseen_medium',53012),'UnseenHard':('unseen_hard',53003)}

def collect(folder):
    digest=json.loads((ROOT/'manifest.json').read_text())['checkpoint_sha256']
    fields=next(csv.reader((ROOT/'evaluation/team_v21/result_template.csv').open()))
    rows=[]
    for label,(name,seed) in LEVELS.items():
        d=json.loads((folder/f'{label}.json').read_text())
        expected={'completed':100,'seed':24,'terrain_seed':seed,'max_episode_steps':960,
                  'observation_dimension':123,'checkpoint_sha256':digest,
                  'task':f'Isaac-Ant-TeamEvalV21-Balance-{label}-v0','reward_protocol':'original_Ant_7_terms_v1'}
        for key,value in expected.items():
            if d.get(key)!=value:raise ValueError(f'{label}: unexpected {key}')
        episodes=d['episodes'];s=d['summary']
        if len(episodes)!=100 or {e['env_id'] for e in episodes}!=set(range(100)):
            raise ValueError('Invalid first-episode population')
        survival=sum(e['steps']==960 and e['timed_out'] and not e['terminated'] for e in episodes)/100
        if not math.isclose(survival,s['timeout_without_failure_rate']):raise ValueError('Survival mismatch')
        if not all(math.isfinite(s[k][stat]) for k in ('reward','steps','forward_displacement_m') for stat in ('mean','std')):
            raise ValueError('Nonfinite result')
        cfg=yaml.load((folder/f'{label}.env.yaml').read_text(),Loader=yaml.BaseLoader)
        weights={k:float(v['weight']) for k,v in cfg['rewards'].items()}
        if weights!=dict(progress=1.,alive=.5,upright=.1,move_to_target=.5,action_l2=-.005,energy=-.05,joint_pos_limits=-.1):
            raise ValueError('Not original Ant rewards')
        if cfg['events']['reset_base']['params']['pose_range']!={}:raise ValueError('Not original reset')
        term=cfg['terminations']['torso_height']
        if not term['func'].endswith(':root_height_below_minimum') or float(term['params']['minimum_height'])!=.31:
            raise ValueError('Not original termination')
        if int(cfg['scene']['terrain']['max_init_terrain_level'])!=4 or float(cfg['episode_length_s'])!=16:
            raise ValueError('Unexpected evaluation configuration')
        rows.append(dict(member='Kim-KiSuk',checkpoint_name=Path(d['checkpoint']).name,checkpoint_sha256=digest,
                         task_id=d['task'],environment=name,seed=24,num_envs=100,obs_dim=123,action_dim=8,
                         reward_mean=s['reward']['mean'],reward_std=s['reward']['std'],
                         steps_mean=s['steps']['mean'],steps_std=s['steps']['std'],survival_rate=survival,
                         forward_mean_m=s['forward_displacement_m']['mean'],forward_std_m=s['forward_displacement_m']['std'],
                         completed_episodes=100,notes=f"v2.1; original rewards/termination/reset; missing_ground={s['missing_ground_episodes']}; missing_scan={s['missing_scan_episodes']}"))
    output=folder/'result_template.csv'
    with output.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    collect(p.parse_args().directory)

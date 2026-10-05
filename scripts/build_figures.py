"""Regenerate figures from published result CSV/JSON; no simulator needed."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/figures'
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((ROOT/'results/team_v21/final/result_template.csv').open()))
names = ['Seen control', 'Unseen easy', 'Unseen medium', 'Unseen hard']
colors = ['#2563eb', '#0891b2', '#d97706', '#7c3aed']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
reward=[float(r['reward_mean']) for r in rows]
axes[0].bar(names,reward,yerr=[float(r['reward_std']) for r in rows],color=colors,capsize=4)
axes[0].set(title='Original Ant reward',ylabel='Episode return (mean ± population SD)',ylim=(0,155))
for i,value in enumerate(reward):axes[0].text(i,4,f'{value:.2f}',ha='center',color='white',weight='bold')
survival=[100*float(r['survival_rate']) for r in rows]
axes[1].bar(names,survival,color=colors)
axes[1].set(title='Survival to 960 steps',ylabel='Episodes without failure (%)',ylim=(0,105))
for i,value in enumerate(survival):axes[1].text(i,value+2,f'{value:.0f}%',ha='center',weight='bold')
for ax in axes:ax.tick_params(axis='x',labelsize=9)
fig.suptitle('Frozen final checkpoint · team v2.1 · seed 24 · 100 first episodes per environment',fontsize=11)
fig.savefig(OUT/'team_v21_results.png',dpi=180);plt.close(fig)

data={d['name']:d for d in json.loads((ROOT/'results/development/summary.json').read_text())}
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,keys,labels,title in [
 (axes[0],['balance4000','balance6000','failure2'],['Balance\n4000','Balance\n6000','Failure −2\n6000'],'E3: original reset'),
 (axes[1],['failure2_eval_lift','final_lift'],['Failure −2\n6000','Final lift\n6000'],'E3: reset +0.15 m for BOTH models')]:
    values=[data[k]['reward']['mean'] for k in keys]
    ax.bar(labels,values,color=['#94a3b8','#2563eb','#0891b2'][:len(keys)])
    for i,(k,v) in enumerate(zip(keys,values)):
        ax.text(i,v+1,f'{v:.2f}\n{100*data[k]["timeout_without_failure_rate"]:.0f}% survival',ha='center',fontsize=9)
    ax.set(title=title,ylabel='Mean original Ant return',ylim=(0,103))
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Development comparisons: separate protocols; one training seed, no causal certainty',fontsize=10)
fig.savefig(OUT/'development_comparison.png',dpi=180);plt.close(fig)

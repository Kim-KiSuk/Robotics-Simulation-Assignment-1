"""Export final TensorBoard learning curves; training return is not evaluation return."""
from pathlib import Path
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
path=next((ROOT/'artifacts/tensorboard/final_lift').glob('events.*'))
acc=EventAccumulator(str(path),size_guidance={'scalars':0});acc.Reload()
reward={e.step:e.value for e in acc.Scalars('Train/mean_reward')}
length={e.step:e.value for e in acc.Scalars('Train/mean_episode_length')}
steps=sorted(reward.keys() & length.keys())
with (ROOT/'results/development/final_training_curve.csv').open('w',newline='') as stream:
    writer=csv.writer(stream,lineterminator="\n");writer.writerow(['iteration','train_mean_reward','train_mean_episode_length'])
    writer.writerows((s,reward[s],length[s]) for s in steps)
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
for ax,source,title in [(axes[0],reward,'Training return (shaped reward)'),(axes[1],length,'Training mean episode length')]:
    values=[source[s] for s in steps]
    average=[sum(values[max(0,i-99):i+1])/min(i+1,100) for i in range(len(values))]
    ax.plot(steps,values,color='#93c5fd',alpha=.4,linewidth=.4,label='Raw')
    ax.plot(steps,average,color='#2563eb',linewidth=1.5,label='Trailing 100 iterations')
    ax.set(title=title,xlabel='Iteration');ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=8)
fig.suptitle('Final Failure2Lift · seed 42 · 1024 environments · scratch training',fontsize=11)
fig.savefig(ROOT/'artifacts/figures/final_training_curve.png',dpi=180)

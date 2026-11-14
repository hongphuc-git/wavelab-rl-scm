"""Paired validation of each checkpoint; truth is used only after rollout."""
import argparse, csv, gc, hashlib, json, sys, os
from pathlib import Path
import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rlscm.backend import SCM
from rlscm.environment import PipeEnv, blur_misfit
from rlscm.paths import LOW,HIGH
from stable_baselines3 import SAC


def metrics(truth,prediction):
    delta=prediction-truth
    ss=((truth-truth.mean(0))**2).sum(0)
    r2=np.full(3,np.nan)
    np.divide((delta**2).sum(0),ss,out=r2,where=ss>0)
    r2=1-r2
    return dict(mae=np.abs(delta).mean(0).tolist(),rmse=np.sqrt((delta**2).mean(0)).tolist(),
                r2=r2.tolist(),mare_percent=(100*np.abs(delta)/np.maximum(np.abs(truth),1e-12)).mean(0).tolist(),
                mean_normalized_error=float((np.abs(delta)/(HIGH-LOW)).mean()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    config=json.loads((args.run/'run_config.json').read_text())
    if config.get('backend','python') == 'python': os.environ['RLSCM_BACKEND']='python'
    torch.set_num_threads(2)
    out=args.run/'evaluation';out.mkdir(exist_ok=True)
    with h5py.File(args.dataset) as f:
        if f.attrs.get('split') != 'validation' or not f['complete'][:].all():
            raise ValueError('Expected complete validation dataset')
        images=f['images'][:];truth=f['parameters'][:]
        ids=f['source_indices'][:] if 'source_indices' in f else np.arange(len(images))
    with h5py.File(config['dataset']) as f:
        if 'source_indices' in f and np.intersect1d(ids,f['source_indices'][:]).size:
            raise ValueError('Train/validation overlap')
    backend=SCM(out/'scm',config.get('nodes',20),config.get('frequencies',256),
                curve_points=config.get('curve_points',256))
    summary=[]
    checkpoints=sorted(args.run.glob('checkpoint_*_steps.zip'),key=lambda x:int(x.stem.split('_')[1]))
    if not checkpoints: raise ValueError('No checkpoints to evaluate')
    try:
        for cp in checkpoints:
            # Avoid allocating the training replay during inference.
            model=SAC.load(cp,device=args.device,buffer_size=1,
                           custom_objects={'learning_rate':1e-4,'lr_schedule':lambda _:1e-4})
            cases=[]
            for i,image in enumerate(images):
                env=PipeEnv(backend,[{'image':image.astype(np.int8)}],image_score=True,
                            horizon=config.get('horizon',8),action_scale=config.get('action_scale',.15),
                            action_decay=config.get('action_decay',1.),
                            score_sigmas=config.get('score_sigmas',[4.,12.]),
                            misfit_mode=config.get('misfit_mode','auto'),param_weight=0.)
                obs,_=env.reset(seed=20261113+i)
                initial=(LOW+env.x*(HIGH-LOW)).copy()
                common_J=blur_misfit(image,env.candidate['image'],(4,12))
                common_prediction=initial.copy()
                history=[dict(parameters=initial.tolist(),J=env.J,J_blur=common_J)]
                for _ in range(env.horizon):
                    obs,*_=env.step(model.predict(obs,deterministic=True)[0])
                    parameters=LOW+env.x*(HIGH-LOW)
                    j=blur_misfit(image,env.candidate['image'],(4,12))
                    if j<common_J: common_J,common_prediction=j,parameters.copy()
                    history.append(dict(parameters=parameters.tolist(),J=env.J,J_blur=j))
                cases.append(dict(case=i,source_index=int(ids[i]),seed=20261113+i,
                                  truth=truth[i].tolist(),initial=initial.tolist(),
                                  final=parameters.tolist(),best_own=env.best_m.tolist(),
                                  best_common=common_prediction.tolist(),J_own=env.best_J,
                                  J_common=common_J,history=history))
            estimates={key:np.asarray([c[key] for c in cases]) for key in ('initial','final','best_own','best_common')}
            row=dict(checkpoint=cp.name,experiment=config.get('misfit_mode','auto'),
                     checkpoint_sha256=hashlib.file_digest(cp.open('rb'),'sha256').hexdigest(),
                     dataset=str(args.dataset),source_indices=ids.tolist(),
                     selection='best_common = minimum blur-NCC J over initialization + rollout',
                     metrics={key:metrics(truth,v) for key,v in estimates.items()},cases=cases)
            (out/(cp.stem+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False))
            with (out/(cp.stem+'.csv')).open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['case','source_index']+[f'{kind}_{name}' for kind in estimates for name in ('h','c0','nu')]+['true_h','true_c0','true_nu'])
                for i,c in enumerate(cases): writer.writerow([i,c['source_index']]+[v for key in estimates for v in c[key]]+c['truth'])
            fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
            for a,name in enumerate(('h (mm)','sqrt(E/rho) (km/s)','nu')):
                ax=axes[a];pred=estimates['best_common'][:,a]
                lo=min(truth[:,a].min(),pred.min());hi=max(truth[:,a].max(),pred.max())
                ax.scatter(truth[:,a],pred,s=18,alpha=.65)
                ax.plot([lo,hi],[lo,hi],'r--',label='Ideal y=x');ax.set_aspect('equal',adjustable='box')
                ax.set(xlabel='True '+name,ylabel='Predicted '+name,
                       title=f"R2={row['metrics']['best_common']['r2'][a]:.3f}; RMSE={row['metrics']['best_common']['rmse'][a]:.4g}")
                ax.grid(alpha=.2)
            fig.suptitle(cp.stem+' | common blur selection | validation (one seed)')
            fig.savefig(out/('scatter_'+cp.stem+'.png'),dpi=180);plt.close(fig)
            summary.append({k:v for k,v in row.items() if k!='cases'})
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print(cp.name,summary[-1]['metrics'],flush=True)
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    finally: backend.close()


if __name__=='__main__':main()

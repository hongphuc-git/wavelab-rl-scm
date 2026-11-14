"""Parallel SAC training on converted bone images for a many-core machine.

Each worker process owns one PipeEnv and one SCM backend, so SCM solves run in
parallel (SubprocVecEnv). The learner runs in the main process, on CUDA when
available. Same environment, J and reward as train_bone_sac_trial.py; only the
throughput and SAC sizing differ.

Example (64-core machine with a GPU):
    python -u scripts/train_bone_sac_parallel.py --n-envs 60 --steps 500000
"""
import os

# One BLAS thread per SCM worker; must be set before numpy is imported so the
# spawned workers inherit it. The learner's thread count is set via torch below.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from rlscm.data_protocol import write_json
from rlscm.model import ENCODERS

IMAGE_BYTES = 3 * 256 * 256  # int8 observation image


class ImageTargets:
    """Lazy h5 reader; opened inside each worker (h5py handles cannot be pickled)."""

    def __init__(self, path, split):
        import h5py
        self.file = h5py.File(path, 'r')
        if self.file.attrs.get('split') != split or not self.file['complete'][:].all():
            self.file.close()
            raise ValueError('Wrong or incomplete dataset split')
        self.images = self.file['images']
        self.parameters = self.file['parameters'][:].astype(float)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {'image': self.images[index].astype(np.int8), 'parameters': self.parameters[index]}

    def close(self):
        self.file.close()


def make_env(rank, folder, dataset, nodes, frequencies, horizon, action_scale, score_sigmas=(), param_weight=0.0,
             param_axis_weights=(1.0, 1.0, 1.0), action_decay=1.0, lookup_table=None, reward_mode="absolute"):
    def factory():
        from rlscm.backend import SCM
        from rlscm.environment import PipeEnv

        class ClosingPipeEnv(PipeEnv):
            def close(self):
                try:
                    self.backend.close()
                finally:
                    self.targets.close()

        worker = folder / f'env_{rank:03d}'
        worker.mkdir(parents=True, exist_ok=True)
        targets = ImageTargets(dataset, 'train')
        if lookup_table:  # E4: nearest precomputed SCM image instead of a solve
            from rlscm.lookup import LookupSCM
            backend = LookupSCM(lookup_table)
        else:
            backend = SCM(worker, nodes, frequencies, curve_points=frequencies)
        return ClosingPipeEnv(backend, targets, horizon=horizon, shuffle_targets=True,
                              image_score=True, action_scale=action_scale,
                              score_sigmas=score_sigmas, param_weight=param_weight,
                              param_axis_weights=param_axis_weights, action_decay=action_decay, reward_mode=reward_mode)
    return factory


class ParallelProgress(BaseCallback):
    def __init__(self, folder, steps, save_every, stop_file=None, target_count=None, replay_every=0):
        super().__init__()
        self.replay_every = replay_every
        self.stopped = False
        self.target_count = target_count
        self.folder, self.steps, self.save_every = folder, steps, save_every
        self.stop_file = stop_file
        self.started = time.perf_counter()
        self.next_save = save_every
        self.window = []
        self.window_nearest = []
        self.window_param = []

    def _on_training_start(self):
        # On resume, the next checkpoint is the next multiple of save_every after the current step.
        self.next_save = (self.num_timesteps // self.save_every + 1) * self.save_every

    def _on_step(self):
        infos = self.locals.get('infos', [])
        self.window.extend(float(i['J']) for i in infos if 'J' in i)
        self.window_nearest.extend(float(i['J_nearest']) for i in infos if 'J_nearest' in i)
        self.window_param.extend(float(i['param_error']) for i in infos if 'param_error' in i)
        if self.n_calls % 20 == 0:
            self.record()
        if self.num_timesteps >= self.next_save:
            self.model.save(self.folder / f'checkpoint_{self.num_timesteps}_steps')
            self.next_save += self.save_every
            if self.replay_every and self.num_timesteps % self.replay_every < self.save_every:
                save_replay(self.model, self.folder)
            if shutil.disk_usage(self.folder).free < 2 * 1024 ** 3:
                print('Stopping safely: less than 2 GiB disk space remains.', flush=True)
                return False
        if self.stop_file and Path(self.stop_file).exists():
            print('Stop file found; stopping.', flush=True)
            self.stopped = True
            return False
        return True

    def record(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        elapsed = time.perf_counter() - self.started
        done = self.num_timesteps - self.model._num_timesteps_at_start
        rate = done / elapsed if elapsed else 0.0
        remaining = max(self.steps - done, 0)

        def value(key):
            number = self.model.logger.name_to_value.get(key)
            return float(number) if number is not None and np.isfinite(number) else None

        try:  # distinct train images seen by all workers (each shuffles the full set independently)
            unique = len(set().union(*self.training_env.get_attr('seen_target_ids')))
        except Exception:
            unique = None
        metric = dict(step=int(self.num_timesteps), elapsed_seconds=elapsed,
                      unique_targets=unique, target_count=self.target_count,
                      env_steps_per_second=rate,
                      eta_hours=(remaining / rate / 3600) if rate else None,
                      mean_J=float(np.mean(self.window)) if self.window else None,
                      mean_J_nearest=float(np.mean(self.window_nearest)) if self.window_nearest else None,
                      mean_param_error=float(np.mean(self.window_param)) if self.window_param else None,
                      actor_loss=value('train/actor_loss'), critic_loss=value('train/critic_loss'),
                      entropy_coefficient=value('train/ent_coef'))
        self.window = []
        self.window_nearest = []
        self.window_param = []
        with (self.folder / 'training_metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metric) + '\n')
        temp = self.folder / 'progress.tmp.json'
        temp.write_text(json.dumps(dict(metric, requested_steps=self.steps), indent=2), encoding='utf-8')
        temp.replace(self.folder / 'progress.json')
        if self.n_calls % 200 == 0:
            print(f"step {metric['step']:>8}  {rate:6.1f} steps/s  ETA {metric['eta_hours'] or 0:6.2f} h  "
                  f"mean J {metric['mean_J']}  param err {metric['mean_param_error']}", flush=True)

    def _on_training_end(self):
        self.record()


def save_replay(model, folder):
    """Keep one replay snapshot (latest), written atomically next to the checkpoints."""
    temp = folder / 'replay_buffer.tmp.pkl'
    model.save_replay_buffer(temp)
    temp.replace(folder / 'replay_buffer.pkl')
    (folder / 'replay_buffer.json').write_text(json.dumps(dict(step=int(model.num_timesteps))), encoding='utf-8')


def _count_targets(path):
    import h5py
    try:
        with h5py.File(path, 'r') as data:
            return int(len(data['images']))
    except (OSError, KeyError):
        return None


def main():
    cores = os.cpu_count() or 2
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=200_000, help='environment steps to add')
    parser.add_argument('--n-envs', type=int, default=max(1, cores - 2), help='parallel SCM workers')
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'data/bone_b8_train_256_compact.h5')
    parser.add_argument('--backend', choices=['python', 'matlab'], default='python')
    parser.add_argument('--nodes', type=int, default=20)
    parser.add_argument('--frequencies', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--action-scale', type=float, default=.15)
    parser.add_argument('--score-sigmas', type=float, nargs='*', default=[4.0, 12.0],
                        help='Gaussian sigmas (pixels) for blur_misfit J; pass --score-sigmas with no value for the old nearest-branch J')
    parser.add_argument('--param-weight', type=float, default=0.0,
                        help='privileged reward: also subtract this weight x mean normalized parameter error '
                             '(train labels only; the policy never observes them)')
    parser.add_argument('--encoder', choices=sorted(ENCODERS), default='small',
                        help='image feature extractor (spatial = ~0.6 M params, keeps an 8x8 grid)')
    parser.add_argument('--share-encoder', action='store_true',
                        help='actor reuses the critic encoder (fewer CNN passes per update)')
    parser.add_argument('--param-axis-weights', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='relative weights of h, c0, nu inside the parameter error (E3)')
    parser.add_argument('--action-decay', type=float, default=1.0,
                        help='step size = action_scale * decay**t (E2); 1.0 keeps it constant')
    parser.add_argument('--lookup-table', type=Path, help='h5 from precompute_table.py: train on nearest SCM images (E4)')
    parser.add_argument('--save-replay', action='store_true',
                        help='save the replay buffer with every checkpoint (latest copy only, ~0.4 GB per 1k transitions); '
                             'resuming from a checkpoint in the same folder then restores it')
    parser.add_argument('--replay-every', type=int, default=0,
                        help='also save the replay buffer at checkpoints every N steps (0 = only on a stop request)')
    parser.add_argument('--max-vram-gb', type=float, help='cap this process on the GPU')
    parser.add_argument('--buffer-size', type=int, default=20_000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-starts', type=int, default=5_000)
    parser.add_argument('--gradient-steps', type=int, default=1,
                        help='updates per vector step (default 1); -1 = one per collected transition')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--lr-final', type=float, default=None, help='Optional final learning rate; linear schedule over this training session')
    parser.add_argument('--entropy-coef', default='auto', help='auto, auto_0.01, or fixed coefficient such as 0.01')
    parser.add_argument('--reward-mode', choices=['absolute','parameter_progress'], default='absolute')
    parser.add_argument('--gamma', type=float, default=.9)
    parser.add_argument('--torch-threads', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=20_000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out', type=Path, help='output folder (default runs/rl_bone_parallel_<time>)')
    parser.add_argument('--resume', type=Path, help='model.zip / checkpoint .zip (replay not restored)')
    parser.add_argument('--save-initial', action='store_true', help='Save untrained checkpoint for paired diagnostics')
    parser.add_argument('--dry-run', action='store_true', help='print sizing and exit')
    args = parser.parse_args()

    if args.save_replay:
        args.replay_every = args.save_every
    ram_gib = args.buffer_size * 2 * IMAGE_BYTES / 2 ** 30
    print(f'Replay buffer ~{ram_gib:.1f} GiB RAM (obs + next_obs, int8). '
          f'Workers: {args.n_envs}. Backend: {args.backend}.', flush=True)
    if args.dry_run:
        return
    if args.backend == 'python':
        os.environ['RLSCM_BACKEND'] = 'python'
    else:
        os.environ.pop('RLSCM_BACKEND', None)

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(args.torch_threads)
    if device == 'cuda' and args.max_vram_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_gb / total))
    out = args.out or ROOT / 'runs' / datetime.now().strftime('rl_bone_parallel_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), out=str(out), dataset=str(args.dataset.resolve()), resume=str(args.resume) if args.resume else None,
                  lookup_table=str(args.lookup_table.resolve()) if args.lookup_table else None,
                  curve_points=args.frequencies,
                  score=('blur_ncc_sigmas_' + '_'.join(f'{x:g}' for x in args.score_sigmas)) if args.score_sigmas
                  else 'symmetric_image_distance',
                  reward=(f'{args.param_weight:g} * (error_before - error_after) - 0.01' if args.reward_mode=='parameter_progress' else '-J - 0.01' + (f' - {args.param_weight:g} * param_error' if args.param_weight else '')),
                  image_storage='int8', device=device, replay_ram_gib=ram_gib,
                  torch_version=torch.__version__)
    resume_reward = json.loads((Path(args.resume).parent/'run_config.json').read_text()).get('reward_mode','absolute') if args.resume else args.reward_mode
    if resume_reward != args.reward_mode: raise ValueError('Cannot resume across reward modes')
    write_json(out / 'run_config.json', config)
    print('OUTPUT:', out, flush=True)

    factories = [make_env(i, out, args.dataset.resolve(), args.nodes, args.frequencies,
                          args.horizon, args.action_scale, tuple(args.score_sigmas), args.param_weight,
                          tuple(args.param_axis_weights), args.action_decay,
                          str(args.lookup_table.resolve()) if args.lookup_table else None, args.reward_mode)
                 for i in range(args.n_envs)]
    vec = SubprocVecEnv(factories, start_method='spawn') if args.n_envs > 1 else DummyVecEnv(factories)
    vec = VecMonitor(vec, filename=str(out / 'monitor'))
    try:
        if args.resume:
            prior=json.loads((Path(args.resume).parent/'run_config.json').read_text())
            if prior.get('reward_mode','absolute') != args.reward_mode:
                raise ValueError('Cannot resume across different reward modes')
        if args.resume:
            model = SAC.load(args.resume, env=vec, device=device, buffer_size=args.buffer_size,
                             batch_size=args.batch_size, learning_starts=args.learning_starts,
                             gradient_steps=args.gradient_steps)
            replay = Path(args.resume).parent / 'replay_buffer.pkl'
            replay_meta = replay.with_name('replay_buffer.json')
            replay_matches = (replay_meta.exists() and json.loads(replay_meta.read_text())['step'] == model.num_timesteps)
            if replay.exists() and replay_matches and Path(args.resume).parent.resolve() == out.resolve():
                model.load_replay_buffer(replay)
                print(f'Resuming at {model.num_timesteps} steps with {model.replay_buffer.size() * model.n_envs} replay transitions.', flush=True)
            else:
                model.learning_starts = model.num_timesteps + args.learning_starts
                print(f'Resuming at {model.num_timesteps} steps; replay missing/mismatched. Refill for {args.learning_starts} transitions before updates.', flush=True)
        else:
            model = SAC('MultiInputPolicy', vec, seed=args.seed, device=device,
                        buffer_size=args.buffer_size, batch_size=args.batch_size,
                        learning_starts=args.learning_starts, learning_rate=args.learning_rate,
                        gamma=args.gamma, ent_coef=(args.entropy_coef if args.entropy_coef.startswith('auto') else float(args.entropy_coef)), train_freq=1, gradient_steps=args.gradient_steps, verbose=0,
                        policy_kwargs=dict(features_extractor_class=ENCODERS[args.encoder], normalize_images=False,
                                           net_arch=[256, 256], share_features_extractor=args.share_encoder))
        if args.resume and args.entropy_coef != 'auto':
            raise ValueError('Entropy experiment requires a fresh run to avoid incompatible resumed optimizer state')
        if args.lr_final is not None:
            if not 0 < args.lr_final <= args.learning_rate:
                raise ValueError('Require 0 < lr-final <= learning-rate')
            # SB3 uses global progress on resume. Map it to this session's interval.
            from functools import partial
            from rlscm.training_schedule import session_linear_lr
            model.lr_schedule = partial(session_linear_lr, initial=args.learning_rate,
                final=args.lr_final, start=model.num_timesteps, end=model.num_timesteps + args.steps)
            model.learning_rate = model.lr_schedule
        if args.save_initial and not args.resume:
            model.save(out / 'checkpoint_0_steps')
        start = time.perf_counter()
        progress = ParallelProgress(out, args.steps, args.save_every, target_count=_count_targets(args.dataset),
                                    stop_file=os.environ.get('RLSCM_STOP_FILE'), replay_every=args.replay_every)
        model.learn(total_timesteps=args.steps, reset_num_timesteps=not bool(args.resume), callback=progress)
        if progress.stopped:  # resumable stop: checkpoint + replay, no model.zip (model.zip marks a finished run)
            model.save(out / f'checkpoint_{model.num_timesteps}_steps')
            save_replay(model, out)
            print(f'STOPPED at {model.num_timesteps} steps; resume from the latest checkpoint.', flush=True)
            return 3
        model.save(out / 'model')
        seen = set().union(*vec.get_attr('seen_target_ids'))
        report = dict(configuration=config, total_timesteps=model.num_timesteps,
                      train_seconds=time.perf_counter() - start,
                      training_unique_targets_seen=len(seen),
                      reward=config['reward'],
                      warning='Training metrics do not establish held-out test accuracy.')
        write_json(out / 'report.json', report)
        print('COMPLETE:', json.dumps(report), flush=True)
        return 0
    finally:
        vec.close()


if __name__ == '__main__':
    sys.exit(main())

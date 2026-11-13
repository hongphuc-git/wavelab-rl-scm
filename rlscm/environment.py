"""Physical misfit and Gymnasium environment."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .paths import LOW,HIGH
from .coordinates import rasterize_physical
from scipy.ndimage import distance_transform_edt,gaussian_filter


def image_misfit(target, candidate):
    """Symmetric same-frequency wavenumber distance with robust saturation d/(d+delta) (Sec. 7.4 & 11.2)."""
    a=np.asarray(target,dtype=bool)
    b=np.asarray(candidate,dtype=bool)
    if a.shape!=b.shape or a.ndim!=2:
        raise ValueError('Curve images must have the same 2D shape')
    if not a.any() or not b.any():
        raise ValueError('Cannot score an empty dispersion image')
    span=float(a.shape[1]-1)
    delta=2.0/span
    da=np.where(b.any(axis=1)[:,None],distance_transform_edt(~b,sampling=(1e6,1.0))/span,1.0)[a]
    db=np.where(a.any(axis=1)[:,None],distance_transform_edt(~a,sampling=(1e6,1.0))/span,1.0)[b]
    return float(0.5*(np.mean(da/(da+delta))+np.mean(db/(db+delta))))

def blur_misfit(target, candidate, sigmas=(12.0,)):
    """1 - normalized cross-correlation of Gaussian-blurred binary curve images, averaged over sigmas.

    Unlike the nearest-branch distance, this stays informative far from the target:
    with many dense branches every target pixel has *some* candidate branch nearby,
    which made image_misfit flat beyond ~10% parameter error. Range [0, 2]; 0 = identical.
    """
    a=np.asarray(target)!=0
    b=np.asarray(candidate)!=0
    if a.shape!=b.shape or a.ndim!=2:
        raise ValueError('Curve images must have the same 2D shape')
    if not a.any() or not b.any():
        raise ValueError('Cannot score an empty dispersion image')
    total=0.0
    for sigma in sigmas:
        x=gaussian_filter(a.astype(float),sigma);y=gaussian_filter(b.astype(float),sigma)
        x-=x.mean();y-=y.mean()
        total+=1.0-float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-12))
    return total/len(sigmas)


def _row_runs(row):
    """Centers (column index) of consecutive bright runs in one image row = one mode each."""
    cols = np.flatnonzero(row)
    if not len(cols):
        return cols.astype(float)
    breaks = np.flatnonzero(np.diff(cols) > 1) + 1
    return np.array([g.mean() for g in np.split(cols, breaks)])


def le2024_misfit(target, candidate, f_band=None):
    """Weighted mean-absolute relative wavenumber misfit of Le et al., JASA 156, 3973 (2024), Eq. (16).

    For every observed point (f_j, k_j) of the target image and every candidate mode K_ij at the
    same frequency: d_ij = |k_j - K_ij| / k_j, weight a_ij = exp(-d_ij), J = (1/N) sum_ij a_ij d_ij.
    No mode numbering is needed. A mode is one bright run in an image row (its center column);
    k_j is the column index (proportional to rad/m, the scale cancels in the ratio). Observed points
    at frequencies where the candidate has no mode count as d = 1. f_band=(lo, hi) as fractions of
    the frequency axis restricts the observations (Minonzio et al. 2015 trust 0.4-1.6 MHz).
    """
    a = np.asarray(target) != 0
    b = np.asarray(candidate) != 0
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('Curve images must have the same 2D shape')
    rows = range(a.shape[0])
    if f_band is not None:
        lo, hi = int(round(f_band[0] * (a.shape[0] - 1))), int(round(f_band[1] * (a.shape[0] - 1)))
        rows = range(lo, hi + 1)
    total, count = 0.0, 0
    for r in rows:
        observed = _row_runs(a[r])
        observed = observed[observed >= 1]  # k_j > 0
        if not len(observed):
            continue
        modes = _row_runs(b[r])
        count += len(observed)
        if not len(modes):
            total += len(observed) * np.exp(-1.0)
            continue
        d = np.abs(observed[:, None] - modes[None, :]) / observed[:, None]
        total += float((d * np.exp(-d)).sum())
    if count == 0:
        raise ValueError('Cannot score an empty dispersion image')
    return total / count


def misfit(target, candidate):
    distances = []
    for k, j in zip(target['kk'], target['bins']):
        roots = candidate['kk'][candidate['bins'] == j]
        distances.append(np.min(np.abs(roots-k))/4000 if len(roots) else 1.)
    if not distances:
        raise ValueError('Empty target')
    d = np.array(distances)
    return float(np.mean(d/(d+2/255)))


class PipeEnv(gym.Env):
    def __init__(self, backend, targets, horizon=8, shuffle_targets=False, image_score=False, action_scale=.05, score_sigmas=(), param_weight=0.0,
                 param_axis_weights=(1.0, 1.0, 1.0), action_decay=1.0):
        super().__init__()
        if len(targets) == 0:
            raise ValueError('At least one target is required')
        if not 0 < action_scale <= 1:
            raise ValueError('action_scale must be in (0, 1]')
        self.backend, self.targets, self.horizon = backend, targets, horizon
        self.action_scale = float(action_scale)
        # () keeps the original nearest-branch image J; e.g. (4, 12) selects blur_misfit.
        self.score_sigmas = tuple(float(x) for x in score_sigmas)
        # Privileged training reward: when > 0 and the target carries its true 'parameters',
        # the reward also penalises the normalized parameter error. Never part of the observation,
        # so a trained policy still needs only the image at test time.
        self.param_weight = float(param_weight)
        # E3: relative weight of (h, c0, nu) inside the parameter error (e.g. (1, 1, 3) favours nu).
        self.param_axis_weights = np.asarray(param_axis_weights, float)
        # E2: step size shrinks geometrically, action_scale * action_decay**t (1.0 = constant).
        if not 0 < action_decay <= 1:
            raise ValueError('action_decay must be in (0, 1]')
        self.action_decay = float(action_decay)
        self.shuffle_targets = shuffle_targets
        self.image_score = image_score
        self._remaining_target_ids = []
        self._last_target_id = None
        self.seen_target_ids = set()
        self.episode_count = 0
        self._restored_sampler = False
        self.action_space = spaces.Box(-1., 1., (3,), np.float32)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(-1, 1, (3, 256, 256), np.int8 if image_score else np.float32),
            'state': spaces.Box(0., 1., (5,), np.float32)})

    def score(self):
        if not self.image_score:
            return misfit(self.target, self.candidate)
        if self.score_sigmas:
            return blur_misfit(self.target_image, self.candidate['image'], self.score_sigmas)
        return image_misfit(self.target_image, self.candidate['image'])

    def param_error(self):
        truth = self.target.get('parameters') if isinstance(self.target, dict) else None
        if truth is None:
            return None
        x_true = (np.clip(np.asarray(truth, float), LOW, HIGH) - LOW) / (HIGH - LOW)
        w = self.param_axis_weights
        return float(np.sum(w * np.abs(self.x - x_true)) / np.sum(w))

    def observe(self):
        image = self.candidate.get('image')
        if image is None:
            image = rasterize_physical(self.candidate['ff'], self.candidate['kk'])[0]
        return {'image': np.stack([self.target_image, image, self.target_image-image]).astype(np.int8 if self.image_score else np.float32),
                'state': np.array([*self.x, min(self.J, 1.0), self.t/self.horizon], np.float32)}

    def reset(self, seed=None, options=None):
        if self._restored_sampler:
            seed=None
            self._restored_sampler=False
        super().reset(seed=seed)
        if self.shuffle_targets:
            if seed is not None:
                self._remaining_target_ids = []
                self._last_target_id = None
            if not self._remaining_target_ids:
                self._remaining_target_ids = self.np_random.permutation(len(self.targets)).tolist()
                # Avoid selecting the same target across two consecutive cycles.
                if len(self.targets) > 1 and self._remaining_target_ids[-1] == self._last_target_id:
                    self._remaining_target_ids[0], self._remaining_target_ids[-1] = (
                        self._remaining_target_ids[-1], self._remaining_target_ids[0])
            target_id = self._remaining_target_ids.pop()
            self._last_target_id = target_id
        else:
            target_id = int(self.np_random.integers(len(self.targets)))
        self.seen_target_ids.add(target_id)
        self.episode_count += 1
        self.target = self.targets[target_id]
        self.target_image = self.target.get('image')
        if self.target_image is None:
            self.target_image = rasterize_physical(self.target['ff'], self.target['kk'])[0]
        self.x = self.np_random.uniform(size=3)
        self.t = 0
        self.candidate = self.backend.solve(LOW+self.x*(HIGH-LOW))
        self.J = self.score()
        self.best_J, self.best_m = self.J, LOW+self.x*(HIGH-LOW)
        return self.observe(), {'target_id': target_id}

    def sampler_state(self):
        return dict(remaining=list(self._remaining_target_ids),last=self._last_target_id,
                    seen=sorted(self.seen_target_ids),episodes=self.episode_count,
                    rng=self.np_random.bit_generator.state,target_count=len(self.targets))

    def restore_sampler(self,state):
        if state['target_count']!=len(self.targets):
            raise ValueError('Sampler target count differs from the resumed training partition')
        self._remaining_target_ids=list(state['remaining'])
        self._last_target_id=state['last']
        self.seen_target_ids=set(state['seen']);self.episode_count=state['episodes']
        self.np_random.bit_generator.state=state['rng']
        self._restored_sampler=True

    def step(self, action):
        step = self.action_scale * self.action_decay ** self.t
        self.x = np.clip(self.x+step*np.asarray(action), 0, 1)
        self.candidate = self.backend.solve(LOW+self.x*(HIGH-LOW))
        self.J = self.score()
        self.t += 1
        if self.J < self.best_J:
            self.best_J, self.best_m = self.J, LOW+self.x*(HIGH-LOW)
        info = {'J': self.J}
        if self.image_score and self.score_sigmas:
            # Monitoring only: the original nearest-branch (f,k) distance, not used in the reward.
            info['J_nearest'] = image_misfit(self.target_image, self.candidate['image'])
        reward = -self.J - .01
        error = self.param_error()
        if error is not None:
            info['param_error'] = error
            reward -= self.param_weight * error
        return self.observe(), reward, False, self.t >= self.horizon, info


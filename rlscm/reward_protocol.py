"""Prevent replay and policy continuation under a different reward definition."""
def validate_resume(prior, current):
    defaults = dict(reward_mode='absolute', param_weight=0.0, misfit_mode='auto',
                    score_sigmas=[4.0,12.0], param_axis_weights=[1.,1.,1.])
    for key, default in defaults.items():
        old, new = prior.get(key, default), current.get(key, default)
        if old != new:
            raise ValueError(f'Cannot resume with changed {key}: {old!r} -> {new!r}; use a fresh run')

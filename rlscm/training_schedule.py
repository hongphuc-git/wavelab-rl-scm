"""Learning rate schedule with explicit resumed-session endpoints."""
def session_linear_lr(progress_remaining, initial, final, start, end):
    if end <= start:
        raise ValueError('end must exceed start')
    current = end * (1.0 - progress_remaining)
    fraction = min(1.0, max(0.0, (current - start) / (end - start)))
    return initial + (final - initial) * fraction

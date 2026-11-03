"""Cooperative deadlines and explicit bounded-search outcomes."""
from contextvars import ContextVar
from functools import wraps
from contextlib import contextmanager
import inspect
import time

_deadline = ContextVar('mapf_deadline', default=float('inf'))


class SearchTimeout(RuntimeError):
    pass


class HorizonExceeded(RuntimeError):
    """No path found within an incomplete horizon; infeasibility is unknown."""


def check_deadline():
    if time.perf_counter() >= _deadline.get():
        raise SearchTimeout('Search deadline reached')


@contextmanager
def deadline_scope(seconds):
    token = _deadline.set(min(_deadline.get(), time.perf_counter()+seconds))
    try:
        yield
    finally:
        _deadline.reset(token)


def bounded_solve(fn):
    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        progress = bound.arguments.get('progress')
        if progress is not None:
            progress.clear()
            progress.update(nodes_expanded=0, forced_split_applied=False)
        timeout = bound.arguments['timeout']
        if timeout < 0:
            raise ValueError('timeout must be nonnegative')
        start = time.perf_counter()
        token = _deadline.set(min(_deadline.get(), start + timeout))
        try:
            from mapf.low_level import reset_astar_call_count
            reset_astar_call_count()
            check_deadline()
            result = fn(*args, **kwargs)
            check_deadline()
            return result
        except (SearchTimeout, HorizonExceeded) as exc:
            from mapf.low_level import get_astar_call_count
            return dict(status='TIMEOUT' if isinstance(exc, SearchTimeout) else 'HORIZON_LIMIT',
                        cost=None, nodes_expanded=None, astar_calls=get_astar_call_count(),
                        runtime=time.perf_counter()-start, detail=str(exc))
        finally:
            _deadline.reset(token)
    return wrapped

"""Exploratory policies: shared candidate pool and explicit one-step teacher.

Teacher bounds are a proxy preference, never labels of future solver runtime.
No teacher quantities are passed to the learned model.
"""
import hashlib
import json
import random
import time
import numpy as np
from mapf.baselines import f_aware_residual_selector, guarded_shortlist
from mapf.cardinal import build_constraint_for_agent, classify_conflicts
from mapf.conflict_graph import compute_h4_and_graph
from mapf.features import FeatureExtractor, FEATURE_NAMES
from mapf.low_level import space_time_astar, compute_bfs_h_table
from mapf.search_limits import check_deadline, deadline_scope, SearchTimeout, HorizonExceeded

POOL_SEED = 'ml-revision-pool-v1-71'


def candidate_pool(candidates, context):
    """Champion first; up to five other conflicts selected before teacher labels.

    Hash depends only on the conflict and frozen seed, deliberately not outcome,
    traversal order or mutable global RNG. Same highest-tier input at deployment.
    """
    if not candidates:
        raise ValueError('Empty candidate tier')
    winner = f_aware_residual_selector(candidates, context)
    def key(cc):
        return hashlib.sha256((POOL_SEED + json.dumps(vars(cc.conflict), sort_keys=True)).encode()).hexdigest()
    others = sorted((c for c in candidates if c is not winner), key=key)
    return [winner] + others[:5]


def teacher_key(label):
    if label['status'] != 'COMPLETE':
        return None
    return tuple(float('inf') if x == '+inf' else x for x in label['priority'])


def immediate_children(cc, context, seconds=None):
    """Ordinary binary split; exact low-level paths, full child CG/MVC, pathmax.

    A fresh local cache is shared by the two children of one candidate only.
    Saved paths support independent constraints/cost verification. Timeout is
    unknown, not infeasibility. Online callers omit seconds and propagate caps.
    """
    began = time.perf_counter(); cpu = time.process_time(); branches = []
    def compute():
        check_deadline()
        grid, starts, goals = context['grid'], context['starts'], context['goals']
        paths, constraints = context['paths'], context['per_agent_constraints']
        htables = context.get('h_tables')
        if htables is None:
            htables = [compute_bfs_h_table(grid, g) for g in goals]
        parent_g = sum(len(p)-1 for p in paths)
        parent_f = parent_g + context['effective_h']
        cache = {}
        for agent in (cc.conflict.agent_i, cc.conflict.agent_j):
            check_deadline()
            cs = build_constraint_for_agent(cc.conflict, agent, constraints[agent])
            path = space_time_astar(grid, starts[agent], goals[agent], cs, h_table=htables[agent])
            if path is None:
                branches.append(dict(agent=agent, status='INFEASIBLE', g=None, raw_h=None, effective_f='+inf'))
                continue
            child_paths = list(paths); child_paths[agent] = path
            child_cs = list(constraints); child_cs[agent] = cs
            classified = classify_conflicts(grid, starts, goals, child_paths, child_cs,
                enable_bypass=False, h_tables=htables, path_cache=cache)
            raw_h, _ = compute_h4_and_graph(classified)
            g = sum(len(p)-1 for p in child_paths)
            branches.append(dict(agent=agent, status='FEASIBLE', g=g, raw_h=raw_h,
                effective_f=max(g+raw_h, parent_f), replanned_path=path))
        check_deadline()
        if all(b['status']=='INFEASIBLE' for b in branches):
            return dict(status='NODE_INFEASIBLE', branches=branches, priority=None)
        values = [float('inf') if b['effective_f']=='+inf' else b['effective_f'] for b in branches]
        return dict(status='COMPLETE', branches=branches,
                    priority=['+inf' if v==float('inf') else v for v in (min(values), max(values))])
    try:
        if seconds is None:
            result = compute()
        else:
            with deadline_scope(seconds):
                result = compute()
    except (SearchTimeout, HorizonExceeded) as exc:
        if seconds is None:
            raise
        result = dict(status='TIME_CAP' if isinstance(exc, SearchTimeout) else 'HORIZON_LIMIT',
                      branches=branches, priority=None)
    result.update(wall_seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-cpu)
    return result


class SavedLinear:
    def __init__(self, path):
        from pathlib import Path
        path = Path(path)
        metadata = json.loads(Path(str(path)+'.metadata.json').read_text())
        if metadata['feature_names'] != FEATURE_NAMES:
            raise ValueError('Feature schema mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['model_sha256']:
            raise ValueError('Model checksum mismatch')
        data = json.loads(path.read_text())
        self.mean = np.asarray(data['mean']); self.scale = np.asarray(data['scale'])
        self.coef = np.asarray(data['coef']); self.intercept = data['intercept']
        if any(v.shape != (len(FEATURE_NAMES),) for v in (self.mean,self.scale,self.coef)):
            raise ValueError('Invalid model dimensions')
        if not all(np.isfinite(v).all() for v in (self.mean,self.scale,self.coef)) or (self.scale<=0).any():
            raise ValueError('Invalid model values')
        if not np.isfinite(self.intercept):
            raise ValueError('Invalid intercept')

    def predict(self, X):
        return ((np.asarray(X)-self.mean)/self.scale) @ self.coef + self.intercept


class RevisionSelector:
    def __init__(self, mode, model=None, seed=71):
        if mode not in ('model_a','model_b','pool_random','teacher'):
            raise ValueError(mode)
        self.mode=mode; self.model=model; self.rng=random.Random(seed)
        self.extractor=None
        self.stats=dict(selector_calls=0, ml_calls=0, teacher_calls=0, candidates_scored=0,
                       pool_seconds=0., feature_seconds=0., inference_seconds=0., teacher_seconds=0.)

    def __call__(self, candidates, context):
        self.stats['selector_calls']+=1; began=time.perf_counter()
        pool = guarded_shortlist(candidates, context, 3)[0] if self.mode=='model_a' else candidate_pool(candidates,context)
        self.stats['pool_seconds']+=time.perf_counter()-began
        if len(pool)==1:
            return pool[0]
        if self.mode=='pool_random':
            return self.rng.choice(pool)
        if self.mode=='teacher':
            self.stats['teacher_calls']+=1; began=time.perf_counter()
            try:
                labels=[immediate_children(c,context) for c in pool]
                # A proven dead split closes this node through normal CBS branching.
                for cc,label in zip(pool,labels):
                    if label['status']=='NODE_INFEASIBLE':
                        return cc
                return pool[max(range(len(pool)),key=lambda i:teacher_key(labels[i]))]
            finally:
                self.stats['teacher_seconds']+=time.perf_counter()-began
        self.stats['ml_calls']+=1; self.stats['candidates_scored']+=len(pool)
        began=time.perf_counter()
        if self.extractor is None:
            self.extractor=FeatureExtractor(context['grid'],context['goals'])
        X=self.extractor.extract_all(pool,context)
        self.stats['feature_seconds']+=time.perf_counter()-began
        began=time.perf_counter(); scores=np.asarray(self.model.predict(X))
        self.stats['inference_seconds']+=time.perf_counter()-began
        if scores.shape != (len(pool),) or not np.isfinite(scores).all():
            raise ValueError('Invalid learned scores')
        return pool[int(np.argmax(scores))]

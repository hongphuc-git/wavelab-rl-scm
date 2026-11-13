"""Deterministic data partitions and compact binary-image replay storage."""
import copy
import hashlib
import json
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer


def partition(parameters, seed=2026):
    """Group identical labels so duplicate physical targets cannot cross splits."""
    values=np.asarray(parameters,dtype='<f8')
    groups={}
    for i,row in enumerate(values):
        groups.setdefault(tuple(row),[]).append(i)
    keys=list(groups)
    if len(keys)<10:
        raise ValueError('At least 10 distinct parameter tuples required for a held-out split')
    order=np.random.default_rng(seed).permutation(len(keys))
    n=len(keys);a=int(.9*n)
    result={name:sorted(i for j in ids for i in groups[keys[j]])
            for name,ids in [('train',order[:a]),('validation',order[a:])]}
    result.update(seed=seed,fingerprint=hashlib.sha256(values.tobytes()).hexdigest(),version=2,test_source='external_reserved_dataset')
    return result


def write_json(path,data):
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)


class CompactReplayBuffer(DictReplayBuffer):
    """Store {-1,0,1} image channels in int8; leave scalar states in float32."""
    def __init__(self,buffer_size,observation_space,*args,**kwargs):
        compact=copy.deepcopy(observation_space)
        compact.spaces['image'].dtype=np.dtype(np.int8)
        super().__init__(buffer_size,compact,*args,**kwargs)
        self.observation_space=observation_space

    def add(self,obs,next_obs,*args,**kwargs):
        for item in [obs,next_obs]:
            image=np.asarray(item['image'])
            if not np.all((image==-1)|(image==0)|(image==1)):
                raise ValueError('Compact replay requires binary images and signed binary residuals')
        return super().add(obs,next_obs,*args,**kwargs)

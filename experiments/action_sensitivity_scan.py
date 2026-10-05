import sys...
import env_common
import numpy as np

def _attr(*names):
    for name in names:
        if not isinstance(name, str):
            continue
        obj = getattr(env_common, name, None)
        if obj is not None:
            return obj
    return None

def _call_loader(loader, model_name):
    try:
        return loader(model_name, frozen=True)
    except TypeError:
        try:
            return loader(model_name)
        except TypeError:
            return loader()

@experiment_main
def run(out_dir: str, args: dict) -> dict:
    args = args or {}
    try:
        model_loader = _attr('load_model', 'load_world_model')
        if model_loader is None:
            return {'ok': False, 'error': 'model loader not found'}
        dataset_loader = _attr('load_dataset')
        stats_getter = _attr('get_action_stats')
        return {'ok': True}
    except Exception as e:
        return {'ok': False, 'error': str(e)}
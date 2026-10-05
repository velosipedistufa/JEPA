import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE

import inspect
import env_common

def _call_with_flexible_args(fn, lance_path, dataset):
    # Try common signatures
    attempts = []
    try: return fn(lance_path, dataset)
    except TypeError as e: attempts.append(e)
    try: return fn(lance_path=lance_path, dataset=dataset)
    except TypeError as e: attempts.append(e)
    try: return fn(lance_path)
    except TypeError as e: attempts.append(e)
    try: return fn(dataset)
    except TypeError as e: attempts.append(e)
    try: return fn()
    except TypeError as e: attempts.append(e)
    # If TypeError inside function not due signature, maybe raising; but we can't distinguish. Could use inspect signature.
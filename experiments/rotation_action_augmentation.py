import env_common

def _get_env_common_attr(names, contains=()):
    for name in names:
        if hasattr(env_common, name):
            obj = getattr(env_common, name)
            if callable(obj): return obj
    for name in dir(env_common):
       lname=name.lower()
       if all(token in lname for token in contains):
          obj = getattr(env_common, name)
          if callable(obj): return obj
    return None
```python
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE
import env_common

@experiment_main
def run(out_dir: str, args: dict) -> dict:
    try:
        args = args or {}
        model_path = args.get("model_path")
        dataset_path = args.get("dataset_path")
        if model_path is None or dataset_path is None:
            return {"ok": False, "error": "model_path and dataset_path are required"}

        required = ("load_model", "load_dataset", "action_stats", "evaluate_cem")
        missing = [name for name in required if not callable(getattr(env_common, name, None))]
        if missing:
            return {"ok": False, "error": f"env_common is missing required helpers: {', '.join(missing)}"}

        model = env_common.load_model(model_path)
        data = env_common.load_dataset(dataset_path)
        stats = env_common.action_stats(data)

        sweep = []
        rotation_scales = args.get("rotation_scales") or []
        translation_scales = args.get("translation_scales") or []

        for rot_scale in rotation_scales:
            for trans_scale in translation_scales:
                try
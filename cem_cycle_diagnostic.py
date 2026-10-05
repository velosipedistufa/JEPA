import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "automation")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from env_common import experiment_main, DEVICE

@experiment_main
def run(out_dir: str, args: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    return {
        "ok": True,
        "device": str(DEVICE),
    }
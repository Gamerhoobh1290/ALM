import json
from pathlib import Path
from adamlm.downloads import download
from adamlm.storage import StorageBudget

manifest = json.loads(Path("config/tinystories-v2.json").read_text())
cfg = json.loads(Path("config/bpe512.json").read_text())["storage"]
guard = StorageBudget(".",cfg["project_budget_gb"],cfg["cache_limit_gb"],cfg["minimum_free_disk_gb"])
for split in ("train","validation"):
    download(manifest,split,guard)

"""Preserve source checkpoints; explicitly reset only the dataset on source change."""
import copy
import json
import os
from pathlib import Path
import torch
from filelock import FileLock
from adamlm.bpe import BPETokenizer
from adamlm.downloads import sha256
from adamlm.local_data import LocalStoriesStream, read_stories, PARSER
from adamlm.storage import StorageBudget
from adamlm.control import write_status

old_run = Path("results/bpe512-run")
new_run = Path("results/bpe512-local-run")
guard = StorageBudget(".")
guard.check(256*2**20)
with FileLock(str(old_run / ".training.lock"),timeout=0):
    source = sorted((old_run / "checkpoints").glob("step_*.pt"))[-1]
    old_hashes = {str(p):sha256(p) for p in (old_run / "checkpoints").glob("step_*.pt")}
    if (new_run / "checkpoints").exists():
        raise SystemExit("Continuation already exists; refusing to overwrite")
    state = torch.load(source,map_location="cpu",weights_only=False)
    manifest = json.loads(Path("config/tinystories-v2.json").read_text())
    val_spec = manifest["validation"]
    val_file = Path("data/tinystories-v2") / val_spec["name"]
    if sha256(val_file) != val_spec["sha256"]:
        raise ValueError("Validation file checksum mismatch")
    val_texts, _ = read_stories(val_file,limit=64)
    snapshot_path = Path("data/tinystories-v2/validation-snapshot.json")
    snapshot_path.write_text(json.dumps(dict(source=manifest, split="validation", parser=PARSER, texts=val_texts),ensure_ascii=False),encoding="utf-8")
    cfg = copy.deepcopy(state["extra"]["bpe_protocol"]["config"])
    cfg["run_dir"] = str(new_run).replace("\\","/")
    cfg["validation"] = str(snapshot_path).replace("\\","/")
    cfg["local_manifest"] = "config/tinystories-v2.json"
    tok = BPETokenizer(cfg["tokenizer"])
    stream = LocalStoriesStream(cfg["local_manifest"],tokenizer=tok,**cfg["data"])
    stream.stats.tokens_consumed = state["dataset"]["stats"]["tokens_consumed"]
    stream.source_start_tokens = stream.stats.tokens_consumed
    record = dict(exact_dataset_resume=False, reason="TinyStoriesV2-GPT4 differs from old dataset viewer corpus",
        parent_checkpoint=str(source),parent_sha256=sha256(source), preserved_step=state["step"],
        preserved_global_tokens=stream.stats.tokens_consumed, old_dataset=state["dataset"],
        new_source=stream.identity,new_byte_offset=0,new_story_offset=0,new_buffer=[],
        original_checkpoint_hashes=old_hashes)
    state["dataset"] = stream.state_dict()
    state["extra"]["source_transition"] = {k:v for k,v in record.items() if k != "old_dataset"}
    state["extra"]["bpe_protocol"] = dict(config=cfg,tokenizer_sha256=tok.sha256,validation_sha256=sha256(snapshot_path))
    new_checkpoints = new_run / "checkpoints"
    new_checkpoints.mkdir(parents=True)
    target = new_checkpoints / source.name
    temporary = target.with_suffix(".tmp")
    torch.save(state,temporary)
    os.replace(temporary,target)
    # Exact tensor/state comparison: data/protocol alone may differ.
    restored = torch.load(target,map_location="cpu",weights_only=False)
    original = torch.load(source,map_location="cpu",weights_only=False)
    def equal(a,b):
        if isinstance(a,torch.Tensor):
            return torch.equal(a,b)
        if isinstance(a,dict):
            return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):
            return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    for key in ("model","optimizer","rng","step","model_config"):
        assert equal(restored[key],original[key]), key
    assert all(sha256(p)==value for p,value in old_hashes.items())
    (new_run / "source-transition.json").write_text(json.dumps(record,indent=2))
    (new_run / "launcher-config.json").write_text(json.dumps(dict(target_tokens=cfg["target_tokens"])))
    Path("config/bpe512-local.json").write_text(json.dumps(cfg,indent=2))
    write_status(new_run,state="ready",step=state["step"],tokens=stream.stats.tokens_consumed,
        target_tokens=cfg["target_tokens"],checkpoint=str(target),checkpoint_step=state["step"],
        detail="Explicit GPT-4 source continuation prepared at byte zero; production training not started")
    print(json.dumps({k:v for k,v in record.items() if k != "old_dataset"},indent=2))

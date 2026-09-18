import sys
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "src")
from adamlm import research

tmp = tempfile.gettempdir()
research.write_provider_config(tmp, {
    'enabled': True,
    'model': 'vendor/primary:free',
    'fallback_models': ['vendor/fallback:free'],
})
research.store_api_key(tmp, 'test-key')

ckpt_dir = Path(tmp) / "results" / "test-run" / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt = ckpt_dir / "step_00000001.pt"
ckpt.write_bytes(b"fake")
(Path(tmp) / "results" / "test-run" / "launcher-config.json").write_text(json.dumps({
    "target_tokens": 100, "stage": "sft", "dataset": "dolly"
}))
(Path(tmp) / "results" / "test-run" / "status.json").write_text(json.dumps({
    "state": "target_reached", "tokens": 100, "target_tokens": 100, "stage": "sft", "dataset": "dolly", "step": 1, "checkpoint_step": 1
}))

session = research.new_session(tmp, max_tokens=100, max_seconds=60, max_api_requests=2, use_ai=True)
sid = session["session_id"]

class DryCtx(research.Ctx):
    def launch(self, *a, **k):
        raise AssertionError("must not reach training in this test")
    def sleep(self, s): pass

class MockResponse:
    def __init__(self, model):
        self.model = model
    def read(self):
        if self.model == 'vendor/primary:free':
            return b'{"error": {"message": "Rate limit exceeded", "code": 429}}'
        # Build the proposal JSON properly using json.dumps
        proposal = {
            "hypothesis": "test",
            "parent_checkpoint": str(ckpt),
            "dataset": "dolly",
            "target_tokens": 100,
            "max_seconds": 60,
            "eval_plan": "test"
        }
        content = json.dumps(proposal)
        response = {"choices": [{"message": {"content": content}}]}
        return json.dumps(response).encode()
    def __enter__(self): return self
    def __exit__(self, *a): pass

def mock_urlopen(request, timeout=None):
    import json as _json
    body = request.data.decode('utf-8') if request.data else '{}'
    try:
        model = _json.loads(body).get('model')
    except Exception:
        model = None
    return MockResponse(model)

from unittest.mock import patch
with patch('urllib.request.urlopen', mock_urlopen):
    session = research.new_session(tmp, max_tokens=100, max_seconds=60, max_api_requests=2, use_ai=True)
    sid = session["session_id"]
    research.start_session(sid, tmp, ctx=research.Ctx())
    current = research.read_session(sid, tmp)
    deadline = time.time() + 30
    while current.get("status") == "running" and time.time() < deadline:
        time.sleep(0.1)
        current = research.read_session(sid, tmp)

print("FINAL STATUS:", current.get("status"))
print("STOP_REASON:", current.get("stop_reason"))
print("EXPS:", current.get("experiments"))
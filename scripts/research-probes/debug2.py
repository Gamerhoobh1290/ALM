import sys
import tempfile
import time
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

session = research.new_session(tmp, max_tokens=100, max_seconds=60,
                               max_api_requests=1, use_ai=True)
sid = session["session_id"]

class DryCtx(research.Ctx):
    def launch(self, *a, **k):
        raise AssertionError("must not reach training in this test")
    def sleep(self, s): pass

call_count = [0]

class MockResponse:
    def __init__(self, model):
        self.model = model
    def read(self):
        call_count[0] += 1
        # First call: return a valid proposal (not stop)
        # Second call: hit the API cap
        if call_count[0] == 1:
            return b'{"choices": [{"message": {"content": "{\\"hypothesis\\": \\"test\\", \\"parent_checkpoint\\": \\"test.pt\\", \\"dataset\\": \\"dolly\\", \\"target_tokens\\": 100, \\"max_seconds\\": 60, \\"eval_plan\\": \\"test\\"}"}}]}'
        return b'{"choices": [{"message": {"content": "{\\"stop\\": true, \\"reason\\": \\"done\\"}"}}]}'
    def __enter__(self): return self
    def __exit__(self, *a): pass

def mock_urlopen(request, timeout=None):
    return MockResponse(None)

with patch('urllib.request.urlopen', mock_urlopen):
    research.start_session(sid, tmp, ctx=DryCtx())
    current = research.read_session(sid, tmp)
    deadline = time.time() + 30
    while current.get("status") == "running" and time.time() < deadline:
        time.sleep(0.1)
        current = research.read_session(sid, tmp)

print("FINAL STATUS:", current.get("status"))
print("STOP_REASON:", current.get("stop_reason"))
print("EXPS:", current.get("experiments"))
print("CALL COUNT:", call_count[0])
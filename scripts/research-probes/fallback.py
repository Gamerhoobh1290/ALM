import sys
import tempfile
import json
from unittest.mock import patch

sys.path.insert(0, "src")
from adamlm import research

tmp = tempfile.gettempdir()
research.write_provider_config(tmp, {
    'enabled': True,
    'model': 'vendor/primary:free',
    'fallback_models': ['vendor/fallback:free', 'vendor/alt:free'],
})
research.store_api_key(tmp, 'test-key')

class MockResponse:
    def __init__(self, model):
        self.model = model
        self.status = 200
    def read(self):
        if self.model == 'vendor/primary:free':
            return b'{"error": {"message": "Rate limit exceeded", "code": 429}}'
        return b'{"choices": [{"message": {"content": "{\\"hypothesis\\": \\"test\\", \\"parent_checkpoint\\": \\"test.pt\\", \\"dataset\\": \\"dolly\\", \\"target_tokens\\": 100, \\"max_seconds\\": 60, \\"eval_plan\\": \\"test\\"}"}}]}'
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass

def mock_urlopen(request, timeout=None):
    model = None
    body = request.data.decode('utf-8') if request.data else '{}'
    try:
        model = json.loads(body).get('model')
    except:
        pass
    return MockResponse(model)

with patch('urllib.request.urlopen', mock_urlopen):
    result = research._chat_completion_with_fallback(
        tmp, messages=[{'role': 'user', 'content': 'test'}]
    )
    print('SUCCESS:', result)
    print('Model used:', result['model'])
    print('Attempts:', result['attempts'])
    print('API requests:', result['api_requests'])
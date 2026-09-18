import hashlib
import json
from pathlib import Path
import pytest
import requests
from adamlm.downloads import download
from adamlm.local_data import LocalStoriesStream, read_stories
from adamlm.storage import StorageBudget


def manifest(data):
    return dict(dataset="roneneldan/TinyStories",revision="a"*40,
                train=dict(name="train.txt",size=len(data),sha256=hashlib.sha256(data).hexdigest()))


class Response:
    def __init__(self,status=200,chunks=(),headers=None):
        self.status_code,self.chunks,self.headers = status,chunks,headers or {}
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}",response=self)
    def iter_content(self,*args):
        for item in self.chunks:
            if isinstance(item,Exception): raise item
            yield item


class Session:
    def __init__(self,responses): self.responses,self.calls=iter(responses),[]
    def get(self,url,**kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def test_429_502_retry_and_reuse(tmp_path):
    data=b"story bytes"
    source=manifest(data)
    session=Session([Response(429,headers={"Retry-After":"3"}),Response(502),Response(chunks=[data])])
    sleeps=[]
    path=download(source,"train",StorageBudget(tmp_path,minimum_free_gb=0),root=tmp_path/'data',session=session,sleep=sleeps.append,notify=lambda s:None)
    assert path.read_bytes()==data and len(session.calls)==3 and sleeps[0]>=3
    assert download(source,"train",StorageBudget(tmp_path,minimum_free_gb=0),root=tmp_path/'data',session=session,notify=lambda s:None)==path
    assert len(session.calls)==3


def test_interrupted_download_resumes_range(tmp_path):
    data=b"123456789"
    session=Session([Response(chunks=[data[:4],requests.Timeout("interrupted")]),
                     Response(206,chunks=[data[4:]],headers={"Content-Range":"bytes 4-8/9"})])
    path=download(manifest(data),"train",StorageBudget(tmp_path,minimum_free_gb=0),root=tmp_path/'data',session=session,sleep=lambda _:None,notify=lambda _:None)
    assert path.read_bytes()==data
    assert session.calls[1]["headers"]["Range"]=="bytes=4-"


def test_local_cursor_buffer_and_no_network(tmp_path,monkeypatch):
    data=b"\nFirst little story.\n<|endoftext|>\nSecond longer story.\n<|endoftext|>\nLast story."
    (tmp_path/'train.txt').write_bytes(data)
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest(data)))
    monkeypatch.setattr(requests.Session,"get",lambda *a,**k:pytest.fail("local reader must never access network"))
    stream=LocalStoriesStream(path,root=tmp_path,page_size=1)
    assert read_stories(tmp_path/'train.txt',limit=3)[0]==["First little story.","Second longer story.","Last story."]
    stream.next_batch(1,8)
    saved=stream.state_dict()
    expected=[stream.next_batch(1,8) for _ in range(3)]
    resumed=LocalStoriesStream(path,root=tmp_path,page_size=1)
    resumed.load_state_dict(saved)
    assert [resumed.next_batch(1,8) for _ in range(3)]==expected
    wrong=dict(saved,local_source={})
    with pytest.raises(ValueError,match="Source changed"):
        resumed.load_state_dict(wrong)


def test_failed_checksum_never_promotes(tmp_path):
    session=Session([Response(chunks=[b"bad!"])])
    with pytest.raises(ValueError,match="SHA-256"):
        download(manifest(b"good"),"train",StorageBudget(tmp_path,minimum_free_gb=0),root=tmp_path/'data',session=session,notify=lambda _:None)
    assert not (tmp_path/'data/train.txt').exists()

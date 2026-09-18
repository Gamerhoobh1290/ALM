"""One-time, bounded, resumable downloads of pinned official files."""
import hashlib
import os
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
import requests
import truststore
from filelock import FileLock

truststore.inject_into_ssl()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def retry_delay(attempt, value=None):
    delay = min(60, 2**attempt) + random.SystemRandom().uniform(0, 1)
    if value:
        try:
            required = float(value)
        except ValueError:
            required = parsedate_to_datetime(value).timestamp()-time.time()
        delay = max(delay, required)
    return delay


def download(manifest, split, guard, root="data/tinystories-v2", session=None, sleep=time.sleep, attempts=5, notify=print):
    root = Path(root).resolve()
    if not root.is_relative_to(guard.root):
        raise ValueError("Dataset directory must stay inside project")
    root.mkdir(parents=True, exist_ok=True)
    spec = manifest[split]
    target = root / spec["name"]
    partial = target.with_suffix(".part")
    session = session or requests.Session()
    with FileLock(str(target)+".lock",timeout=0):
        if target.is_symlink() or partial.is_symlink():
            raise ValueError("Dataset symlinks are not allowed")
        if target.exists():
            if target.stat().st_size != spec["size"] or sha256(target) != spec["sha256"]:
                raise ValueError("Existing dataset failed integrity verification; left untouched")
            notify(f"Verified reusable {target.name}")
            return target
        url = f'https://huggingface.co/datasets/{manifest["dataset"]}/resolve/{manifest["revision"]}/{spec["name"]}'
        for attempt in range(attempts):
            retry_after = None
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > spec["size"]:
                    raise ValueError("Oversized partial file; left untouched for inspection")
                guard.check(spec["size"]-offset + 192*2**20)
                if offset < spec["size"]:
                    notify(f"Downloading {target.name} from byte {offset} of {spec['size']}")
                    headers = {"Accept-Encoding":"identity"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    with session.get(url,headers=headers,stream=True,timeout=(15,30)) as response:
                        retry_after = response.headers.get("Retry-After")
                        response.raise_for_status()
                        if response.status_code == 206:
                            if not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                                raise requests.RequestException("Invalid download range")
                        else:
                            offset = 0
                        with partial.open("ab" if offset else "wb") as handle:
                            for chunk in response.iter_content(1024*1024):
                                if handle.tell()+len(chunk) > spec["size"]:
                                    raise ValueError("Remote file exceeded published size")
                                guard.check_free_disk(192*2**20 + len(chunk))
                                handle.write(chunk)
                if partial.stat().st_size != spec["size"]:
                    raise requests.RequestException("Incomplete download")
                if sha256(partial) != spec["sha256"]:
                    raise ValueError("Partial file SHA-256 mismatch; preserved, not promoted")
                os.replace(partial,target)
                notify(f"SHA-256 verified: {target.name}")
                return target
            except requests.RequestException as exc:
                if isinstance(exc,requests.HTTPError) and exc.response.status_code not in (408,429,500,502,503,504):
                    raise
                if attempt+1 == attempts:
                    raise
                delay = retry_delay(attempt,retry_after)
                if delay > 300:
                    raise requests.RequestException(f"Server requests {delay:.0f}s cooldown; retry later") from exc
                notify(f"Network retry {attempt+2}/{attempts} after {delay:.1f}s: {exc}")
                sleep(delay)

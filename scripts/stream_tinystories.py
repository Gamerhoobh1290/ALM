"""Read a few TinyStories examples and show the bounded request behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from adamlm.data import TinyStoriesStream


stream = TinyStoriesStream(page_size=2)
x, y = stream.next_batch(batch_size=1, block_size=64)
print(f"stories_read={stream.stats.stories_read} requests={stream.stats.requests} next_offset={stream.next_offset}")
print(f"tokens_in_example={len(x[0])} first_token_ids={x[0][:12]} target_ids={y[0][:12]}")
print("full_dataset_downloaded=False")

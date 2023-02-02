import prior
import gzip
import wget
import urllib.request
import os
from tqdm import tqdm

from prior import LazyJsonDataset


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    for split, size in [("train", 100_000)]:
        if not f"{split}.jsonl.gz" in os.listdir('./'):
                url = "https://anonymous-neurips22.s3.us-west-2.amazonaws.com/a4h/train.jsonl.gz"
                urllib.request.urlretrieve(url,'./{}.jsonl.gz'.format(split))
                
        with gzip.open(f"{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-100k", split=split
        )
    return prior.DatasetDict(**data)

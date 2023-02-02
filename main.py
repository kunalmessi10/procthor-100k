import prior
import gzip
import wget
import os
from tqdm import tqdm

from prior import LazyJsonDataset


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    for split, size in [("train", 100_000)]:
        if not f"{split}.jsonl.gz" in os.listdir('./'):
                print (os.listdir('./'))
                wget.download("https://anonymous-neurips22.s3.us-west-2.amazonaws.com/a4h/train.jsonl.gz")
        with gzip.open(f"{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-100k", split=split
        )
    return prior.DatasetDict(**data)

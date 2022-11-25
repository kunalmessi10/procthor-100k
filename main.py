

import gzip

from tqdm import tqdm

import prior

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")

from utils.custom_lazy_json_dataset import NoCacheLazyJsonDataset


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    for split, size in [("train", 100_000)]:
        with gzip.open(f"{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = NoCacheLazyJsonDataset(
            data=houses, dataset="procthor-100k", split=split
        )
    return prior.DatasetDict(**data)

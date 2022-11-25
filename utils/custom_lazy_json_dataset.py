import json
from typing import Any, Dict, List, Optional, Sequence, Union

# Literal was introduced in Python 3.8
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from prior import Dataset


class NoCacheLazyJsonDataset(Dataset):
    """Lazily load the json house data without caching."""

    def __init__(
        self, data: List[Any], dataset: str, split: Literal["train", "val", "test"]
    ) -> None:
        super().__init__(data, dataset, split)

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        sample = json.loads(self.data[index])
        return sample
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return super().__len__()

    def __repr__(self):
        """Return a string representation of the dataset."""
        return super().__repr__()

    def __str__(self):
        """Return a string representation of the dataset."""
        return super().__str__()


    def __iter__(self):
        """Return an iterator over the dataset."""
        for i, x in enumerate(self.data):
            sample = json.loads(x)
            yield sample

    def select(self, indices: Sequence[int]) -> "Dataset":
        """Return a new dataset containing only the given indices."""
        # ignoring type checker due to mypy bug with attrs
        return NoCacheLazyJsonDataset(
            data=[self.data[i] for i in indices],
            dataset=self.dataset,
            split=self.split,
        )  # type: ignore


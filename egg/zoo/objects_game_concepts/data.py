from __future__ import annotations
from typing import List

import json
import random
import torch
import torch.utils.data
from dataclasses import dataclass


@dataclass
class Task:
    # The total number of feature values.
    num_features: int
    # The number of items in context (num_distractors + 1).
    num_context: int


@dataclass
class Dialog:
    """This class captures the interaction between agents (as to be processed).

    The raw data is in json format. This representation is agnostic to speaking vs.
    listening and will need to be further processed (otherwise the model will see
    forbidden information).

    NOTE: The utterances are place-holders and determine only the number of words generated.
    """

    context: torch.Tensor  # [items x dim] context
    referent: torch.Tensor  # [1 x dim] referent (input)
    target: int  # [1] referent (label)


def collate(batch: List[Dialog]):
    """Collate is responsible for organizing and merging together symmetricple dialogs.

    Parameters
    ----------
    batch : ``List[Dialog]``
        List of all dialogs in the batch.

    Attributes
    ----------
    contexts : ``torch.Tensor``  # [batch x items x dim]
        The context.
    targets : ``torch.Tensor``  # [batch x 1 x dim]
        The referent (features).
    labels : ``torch.LongTensor``  # [batch x 1]
        The referent (label).
    """
    contexts = torch.stack([d.context for d in batch])
    targets = torch.stack([d.referent for d in batch]).squeeze()
    labels = torch.stack([d.target for d in batch]).squeeze()
    return targets, labels, contexts


class ToTensor:
    """Converts an item to tensor. """

    def __init__(self, features: List[str]):
        features = list(sorted(features))
        self.num_features = len(features)
        self.dict = {u: i for i, u in enumerate(features)}

    def __call__(self, item: List[str]) -> torch.tensor:
        out = torch.zeros(self.num_features, requires_grad=False)
        for feature in item:
            out[self.dict[feature]] += 1
        return out


@dataclass
class Dataset(torch.utils.data.Dataset):
    to_tensor: ToTensor
    items: List[List[str]]
    num_context: int

    @classmethod
    def build(
        cls,
        items: List[List[str]],
        features: List[str],
        # max_num_words: int,
        # num_vocab: int,
        num_context: int,
    ) -> Dataset:
        to_tensor = ToTensor(features)
        return Dataset(to_tensor, items, num_context) # max_num_words, num_vocab,

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, key) -> Dialog:
        return self._sample_dialog(key)

    def _sample_dialog(self, key) -> Dialog:
        # Sample K - 1, unique (and different from the referent) items.
        referent = self.items[key]
        distractors = random.sample(
            self.items[:key] + self.items[key + 1 :], self.num_context - 1
        )

        # Shuffle the items and maintain the original referent's index.
        _context = list(enumerate([referent] + distractors))
        random.shuffle(_context)
        indices, context_text = zip(*_context)
        r = [j for j, i in enumerate(indices) if i == 0][0]

        # Construct Dialog instance.
        target = torch.tensor([r])
        context = torch.stack([self.to_tensor(i) for i in context_text])
        referent = context[target]
        return Dialog(context, referent, target)


def get_dataloader(
    items: List[List[str]],
    features: List[str],
    task: Task,
    batch_size: int,
    shuffle: bool,
):
    """Build and return the dataloader with the appropriate dataset."""
    dataset = Dataset.build(
        items,
        features,
        num_context=task.num_context,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        drop_last=False,
    )


def load_dataloader(
    dataset_path: str, dataset: str, task: Task, features, batch_size: int = 64
) -> torch.utils.data.DataLoader:
    with open(f"{dataset_path}/{dataset}.json", "r") as jf:
        d = json.load(jf)
        return get_dataloader(
            d, features, task, batch_size, shuffle=True)

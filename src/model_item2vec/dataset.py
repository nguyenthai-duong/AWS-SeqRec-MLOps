import json
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.distributed import get_rank, get_world_size
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm


class SkipGramDataset(IterableDataset):
    """Dataset for training a SkipGram model using sequences of item indices."""

    def __init__(
        self,
        sequences_fp: str,
        interacted: defaultdict = defaultdict(set),
        item_freq: defaultdict = defaultdict(int),
        window_size: int = 2,
        negative_samples: int = 5,
        id_to_idx: dict = None,
        ddp: bool = False,
    ):
        """Initialize the SkipGram dataset.

        Args:
            sequences_fp (str): File path to sequences of item indices in JSONL format.
            interacted (defaultdict[set]): Dictionary tracking items that co-occur in the same basket, ignored during negative sampling.
            item_freq (defaultdict[int]): Dictionary tracking item frequencies for negative sampling.
            window_size (int): Size of the context window for positive pair generation.
            negative_samples (int): Number of negative samples per positive pair.
            id_to_idx (dict, optional): Mapping from item IDs (str) to indices (int). Defaults to None.
            ddp (bool): Whether to use Distributed Data Parallel (DDP) for training. Defaults to False.

        Note:
            The `interacted` and `item_freq` parameters allow reuse of training set data for validation set negative sampling.
        """
        if not sequences_fp.endswith(".jsonl"):
            raise ValueError("sequences_fp must be a .jsonl file")
        self.sequences_fp = sequences_fp
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.ddp = ddp

        # Initialize item ID to index mappings
        self.id_to_idx = dict() if id_to_idx is None else id_to_idx
        self.idx_to_id = (
            dict() if id_to_idx is None else {v: k for k, v in id_to_idx.items()}
        )

        self.interacted = deepcopy(interacted)
        self.item_freq = deepcopy(item_freq)
        self.num_targets = 0  # Total number of items across all sequences

        # Process sequences to build interaction data
        logger.info("Processing sequences to build interaction data...")
        self.sequences = []
        with open(self.sequences_fp, "r") as f:
            for line in tqdm(f, desc="Building interactions"):
                seq = json.loads(line)
                self.sequences.append(seq)

                for item in seq:
                    idx = self.id_to_idx.get(item)
                    if idx is None:
                        idx = len(self.id_to_idx)
                        self.id_to_idx[item] = idx
                        self.idx_to_id[idx] = item
                    self.num_targets += 1

                seq_idx_set = {self.id_to_idx[id_] for id_ in seq}
                for idx in seq_idx_set:
                    self.interacted[idx].update(seq_idx_set)
                    self.item_freq[idx] += 1

        self.num_sequences = len(self.sequences)

        # Determine vocabulary size
        self.vocab_size = (
            len(self.item_freq) if id_to_idx is None else len(id_to_idx)
        )

        # Prepare item frequency array for negative sampling
        items, frequencies = zip(*self.item_freq.items())
        self.item_freq_array = np.zeros(self.vocab_size)
        self.item_freq_array[np.array(items)] = frequencies
        self.items = np.arange(self.vocab_size)

        # Compute smoothed sampling probabilities
        self.sampling_probs = self.item_freq_array**0.75
        self.sampling_probs /= self.sampling_probs.sum()

    def get_process_info(self) -> tuple[int, int]:
        """Retrieve process information for data splitting in distributed training.

        Returns:
            tuple[int, int]: Number of replicas and rank of the current process.
        """
        if not self.ddp:
            return 1, 0

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        world_size = get_world_size()
        process_rank = get_rank()

        num_replicas = num_workers * world_size
        rank = process_rank * num_workers + worker_id
        return num_replicas, rank

    def __iter__(self):
        """Iterate over the dataset, yielding items for the current process.

        Yields:
            dict: A dictionary containing target items, context items, and labels.
        """
        num_replicas, rank = self.get_process_info()
        idx = 0
        for seq in self.sequences:
            for i in range(len(seq)):
                if idx % num_replicas != rank:
                    idx += 1
                    continue
                yield self._get_item(seq, i)
                idx += 1

    def _get_item(self, sequence: list, i: int) -> dict:
        """Generate positive and negative pairs for a given item in a sequence.

        Args:
            sequence (list): List of item IDs in the sequence.
            i (int): Index of the target item in the sequence.

        Returns:
            dict: Dictionary containing tensors for target items, context items, and labels.
        """
        sequence = [self.id_to_idx[item] for item in sequence]
        target_item = sequence[i]

        positive_pairs = []
        labels = []

        # Generate positive pairs within the window
        start = max(i - self.window_size, 0)
        end = min(i + self.window_size + 1, len(sequence))
        for j in range(start, end):
            if i != j:
                context_item = sequence[j]
                positive_pairs.append((target_item, context_item))
                labels.append(1)

        # Generate negative samples
        negative_pairs = []
        for target_item, _ in positive_pairs:
            negative_sampling_probs = deepcopy(self.sampling_probs)
            negative_sampling_probs[list(self.interacted[target_item])] = 0
            if negative_sampling_probs.sum() == 0:
                negative_sampling_probs = np.ones(len(negative_sampling_probs))
            negative_sampling_probs /= negative_sampling_probs.sum()

            negative_items = np.random.choice(
                self.items,
                size=self.negative_samples,
                p=negative_sampling_probs,
                replace=False,
            )
            for negative_item in negative_items:
                negative_pairs.append((target_item, negative_item))
                labels.append(0)

        # Combine pairs and convert to tensors
        pairs = positive_pairs + negative_pairs
        target_items = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
        context_items = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            "target_items": target_items,
            "context_items": context_items,
            "labels": labels,
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate a batch of items into a single dictionary of tensors.

        Args:
            batch (list[dict]): List of dictionaries containing target items, context items, and labels.

        Returns:
            dict: Collated dictionary with concatenated tensors.
        """
        target_items = []
        context_items = []
        labels = []
        for record in batch:
            target_items.append(record["target_items"])
            context_items.append(record["context_items"])
            labels.append(record["labels"])
        return {
            "target_items": torch.cat(target_items, dim=0),
            "context_items": torch.cat(context_items, dim=0),
            "labels": torch.cat(labels, dim=0),
        }

    def save_id_mappings(self, filepath: str) -> None:
        """Save item ID to index mappings to a file.

        Args:
            filepath (str): Path to save the mappings as a JSON file.
        """
        with open(filepath, "w") as f:
            json.dump(
                {
                    "id_to_idx": self.id_to_idx,
                    "idx_to_id": self.idx_to_id,
                },
                f,
            )

    @classmethod
    def get_default_loss_fn(cls) -> nn.Module:
        """Retrieve the default loss function for training.

        Returns:
            nn.Module: Binary Cross Entropy loss function.
        """
        return nn.BCELoss()

    @classmethod
    def forward(cls, model, batch_input: dict, loss_fn: nn.Module = None, device: str = "cpu") -> torch.Tensor:
        """Perform a forward pass and compute the loss for a batch.

        Args:
            model: The SkipGram model.
            batch_input (dict): Batch containing target items, context items, and labels.
            loss_fn (nn.Module, optional): Loss function to use. Defaults to None, in which case BCELoss is used.
            device (str): Device to perform computations on. Defaults to "cpu".

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        predictions = model.predict_train_batch(batch_input, device=device).squeeze()
        labels = batch_input["labels"].float().to(device).squeeze()
        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()
        return loss_fn(predictions, labels)
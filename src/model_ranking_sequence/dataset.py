import json
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.distributed import get_rank, get_world_size
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from tqdm.auto import tqdm


class UserItemRatingDFDataset(Dataset):
    def __init__(
        self,
        df,
        user_col: str,
        item_col: str,
        rating_col: str,
        timestamp_col: str,
        item_feature=None,
    ):
        self.df = df.assign(**{rating_col: df[rating_col].astype(np.float32)})
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.item_feature = item_feature

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.df[self.user_col].iloc[idx]
        item = self.df[self.item_col].iloc[idx]
        rating = self.df[self.rating_col].iloc[idx]
        item_sequence = []
        if "item_sequence" in self.df:
            item_sequence = self.df["item_sequence"].iloc[idx]
        item_sequence_ts_bucket = []
        if "item_sequence_ts_bucket" in self.df:
            item_sequence_ts_bucket = self.df["item_sequence_ts_bucket"].iloc[idx]
        item_feature = []
        if self.item_feature is not None:
            item_feature = self.item_feature[idx]
        return dict(
            user=torch.as_tensor(user),
            item=torch.as_tensor(item),
            rating=torch.as_tensor(rating),
            item_sequence=torch.tensor(item_sequence, dtype=torch.long),
            item_sequence_ts_bucket=torch.tensor(
                item_sequence_ts_bucket, dtype=torch.long
            ),
            item_feature=(
                torch.as_tensor(item_feature) if item_feature is not None else []
            ),
        )


class UserItemBinaryDFDataset(UserItemRatingDFDataset):
    def __init__(
        self,
        df,
        user_col: str,
        item_col: str,
        rating_col: str,
        timestamp_col: str,
        item_feature=None,
    ):
        self.df = df.assign(**{rating_col: df[rating_col].gt(0).astype(np.float32)})
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.item_feature = item_feature


class SkipGramDataset(IterableDataset):
    """
    This class represents a dataset for training a SkipGram model.
    """

    def __init__(
        self,
        sequences_fp: str,
        interacted=defaultdict(set),
        item_freq=defaultdict(int),
        window_size=2,
        negative_samples=5,
        id_to_idx=None,
        ddp=False,
    ):
        """
        Args:
            sequences_fp (str): File path to the sequences of item indices in jsonl format.
            interacted_dict (defaultdict(set)): A dictionary that keeps track of the other items that shared the same basket with the target item. Those items are ignored when negative sampling.
            item_freq (defaultdict(int)): A dictionary that keeps track the item frequency. It's used to
            window_size (int): The context window size.
            negative_samples (int): Number of negative samples for each positive pair.
            id_to_idx (dict): Mapper between item id (string) to item index (int)
            ddp (bool): whether we're using DDP for distributed training or not

        The reason that interacted_dict and item_freq can be passed into the initialization is that at val dataset creation we want to do negative sampling based on the data from the train set as well.
        """

        assert sequences_fp.endswith(".jsonl")
        self.sequences_fp = sequences_fp
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.ddp = ddp

        # Convert the input IDs into sequence integer for easier processing
        if id_to_idx is None:
            self.id_to_idx = dict()
            self.idx_to_id = dict()
        else:
            self.id_to_idx = id_to_idx
            self.idx_to_id = {v: k for k, v in id_to_idx.items()}

        self.interacted = deepcopy(interacted)
        self.item_freq = deepcopy(item_freq)
        self.num_targets = 0  # Counter for number of items in all sequences

        # Keep tracked of which item-pair co-occur in one basket
        # When doing negative sampling we do not consider the other items that the target item has shared basket
        logger.info("Processing sequences to build interaction data...")
        self.sequences = []
        with open(self.sequences_fp, "r") as f:
            seq_idx = 0
            # Wrap the file with tqdm to show progress
            for line in tqdm(f, desc="Building interactions"):
                # Parse each line into a Python object
                seq = json.loads(line)
                self.sequences.append(seq)

                for item in seq:
                    idx = self.id_to_idx.get(item)
                    if idx is None:
                        idx = len(self.id_to_idx)
                        self.id_to_idx[item] = idx
                        self.idx_to_id[idx] = item
                    self.num_targets += 1

                seq_idx_set = set([self.id_to_idx[id_] for id_ in seq])
                for idx in seq_idx_set:
                    # An item can be considered that it has interacted with itself
                    # This helps with negative sampling later
                    self.interacted[idx].update(seq_idx_set)
                    self.item_freq[idx] += 1

                seq_idx += 1

        self.num_sequences = seq_idx + 1

        # Total number of unique items
        if id_to_idx is None:
            self.vocab_size = len(self.item_freq)
        else:
            # Need to check this because sometimes the id_to_idx can have more items than the item_freq
            # For example quen previously we filter out sequence length = 1 so there might be some items
            # are excluded
            self.vocab_size = len(id_to_idx)

        # Create a list of items and corresponding probabilities for sampling
        items, frequencies = zip(*self.item_freq.items())
        self.item_freq_array = np.zeros(self.vocab_size)
        self.item_freq_array[np.array(items)] = frequencies

        self.items = np.arange(self.vocab_size)

        # Use a smoothed frequency distribution for negative sampling
        # The smoothing factor (0.75) can be tuned
        self.sampling_probs = self.item_freq_array**0.75
        self.sampling_probs /= self.sampling_probs.sum()

    def get_process_info(self):
        """
        Get information about which process is processing the data so that we can correctly split up the data based on iteration
        """
        if not self.ddp:
            num_replicas = 1
            rank = 0
            return num_replicas, rank

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        world_size = get_world_size()
        process_rank = get_rank()

        num_replicas = num_workers * world_size
        rank = process_rank * num_workers + worker_id

        return num_replicas, rank

    def __iter__(self):
        num_replicas, rank = self.get_process_info()
        idx = 0
        for seq in self.sequences:
            for i in range(len(seq)):
                if idx % num_replicas != rank:
                    idx += 1
                    continue

                yield self._get_item(seq, i)
                idx += 1

    def _get_item(self, sequence, i):
        sequence = [self.id_to_idx[item] for item in sequence]
        target_item = sequence[i]

        positive_pairs = []
        labels = []

        start = max(i - self.window_size, 0)
        end = min(i + self.window_size + 1, len(sequence))

        for j in range(start, end):
            if i != j:
                context_item = sequence[j]
                positive_pairs.append((target_item, context_item))
                labels.append(1)  # Positive label

        # Generate negative samples based on item frequency
        negative_pairs = []

        for target_item, _ in positive_pairs:
            # Mask out the items that the target item has interacted with
            # Then sample the remaining items based on the item frequency as negative items
            negative_sampling_probs = deepcopy(self.sampling_probs)
            negative_sampling_probs[list(self.interacted[target_item])] = 0
            if negative_sampling_probs.sum() == 0:
                # This target_item has interacted with every other items
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

        # Combine positive and negative pairs
        pairs = positive_pairs + negative_pairs

        # Convert to tensor
        target_items = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
        context_items = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            "target_items": target_items,
            "context_items": context_items,
            "labels": labels,
        }

    def collate_fn(self, batch):
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

    def save_id_mappings(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(
                {
                    "id_to_idx": self.id_to_idx,
                    "idx_to_id": self.idx_to_id,
                },
                f,
            )

    @classmethod
    def get_default_loss_fn(cls):
        loss_fn = nn.BCELoss()
        return loss_fn

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        predictions = model.predict_train_batch(batch_input, device=device).squeeze()
        labels = batch_input["labels"].float().to(device).squeeze()

        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()

        loss = loss_fn(predictions, labels)
        return loss

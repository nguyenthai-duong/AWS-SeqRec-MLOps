import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


from model_item2vec.dataset import SkipGramDataset
from model_item2vec.model import SkipGram
from model_item2vec.trainer import LitSkipGram
from model_item2vec.main import train_func


@pytest.fixture
def sample_sequences(tmp_path):
    """Create a temporary JSONL file with sample sequences."""
    sequences = [["item1", "item2"], ["item2", "item3"]]
    sequences_fp = tmp_path / "sequences.jsonl"
    with open(sequences_fp, "w") as f:
        for seq in sequences:
            f.write(json.dumps(seq) + "\n")
    return str(sequences_fp)


@pytest.fixture
def sample_id_mapper():
    """Create a mock IDMapper with item-to-index mappings."""
    idm = MagicMock()
    idm.item_to_index = {"item1": 0, "item2": 1, "item3": 2}
    idm.load = MagicMock(return_value=idm)
    return idm


def test_skipgram_dataset(sample_sequences, sample_id_mapper):
    """Test SkipGramDataset initialization and data iteration."""
    dataset = SkipGramDataset(
        sequences_fp=sample_sequences,
        window_size=1,
        negative_samples=1,
        id_to_idx=sample_id_mapper.item_to_index,
    )
    assert dataset.vocab_size == 3
    assert len(dataset.id_to_idx) == 3

    batch = next(iter(dataset))
    assert "target_items" in batch
    assert "context_items" in batch
    assert "labels" in batch
    assert batch["target_items"].shape == batch["context_items"].shape


def test_skipgram_model():
    """Test SkipGram model forward pass."""
    model = SkipGram(num_items=3, embedding_dim=4)
    target_items = torch.tensor([0, 1])
    context_items = torch.tensor([1, 2])
    probs = model(target_items, context_items)
    assert probs.shape == (2,)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_lit_skipgram_training_step():
    """Test LitSkipGram training step."""
    model = SkipGram(num_items=3, embedding_dim=4)
    lit_model = LitSkipGram(model)
    batch = {
        "target_items": torch.tensor([0, 1]),
        "context_items": torch.tensor([1, 2]),
        "labels": torch.tensor([1.0, 0.0]),
    }
    loss = lit_model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0



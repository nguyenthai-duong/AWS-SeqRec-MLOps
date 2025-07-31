import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from model_ranking_sequence.dataset import UserItemBinaryDFDataset
from model_ranking_sequence.model import Ranker
from model_ranking_sequence.trainer import LitRanker

@pytest.fixture
def sample_df():
    data = {
        "user_id": [0, 1, 0],
        "item_id": [0, 1, 2],
        "rating": [4, 5, 3],
        "timestamp": [1000, 1001, 1002],
        "item_sequence": [[3, 4], [4, 3], [2, 3]],
        "item_sequence_ts_bucket": [[1, 2], [2, 3], [3, 4]],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_item_features():
    return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)

@pytest.fixture
def tmp_path_for_idm(tmp_path):
    idm_path = tmp_path / "id_mapper.pkl"
    idm = MagicMock()
    idm.user_to_index = {1: 0, 2: 1}
    idm.item_to_index = {10: 0, 20: 1, 30: 2, 9: 3, 8: 4, 19: 5, 18: 6, 29: 7, 28: 8}
    idm.get_user_index = lambda x: idm.user_to_index.get(x, 0)
    idm.get_item_index = lambda x: idm.item_to_index.get(x, 0)
    idm.load = MagicMock(return_value=idm)
    with open(idm_path, "wb") as f:
        import dill
        dill.dump(idm, f)
    return str(idm_path)

# Test UserItemBinaryDFDataset
def test_user_item_binary_df_dataset(sample_df, sample_item_features):
    dataset = UserItemBinaryDFDataset(
        df=sample_df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        timestamp_col="timestamp",
        item_feature=sample_item_features,
    )
    
    assert len(dataset) == 3
    assert dataset.df["rating"].dtype == np.float32
    assert all(dataset.df["rating"].isin([0, 1]))
    
    item = dataset[0]
    assert item["user"].item() == 0
    assert item["item"].item() == 0
    assert item["rating"].item() == 1.0
    assert torch.equal(item["item_sequence"], torch.tensor([3, 4], dtype=torch.long))
    assert torch.equal(item["item_feature"], torch.tensor([0.1, 0.2], dtype=torch.float32))

# Test Ranker model
def test_ranker_forward():
    num_users = 3
    num_items = 5
    embedding_dim = 16
    item_sequence_ts_bucket_size = 10
    bucket_embedding_dim = 8
    item_feature_size = 2
    
    model = Ranker(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        item_sequence_ts_bucket_size=item_sequence_ts_bucket_size,
        bucket_embedding_dim=bucket_embedding_dim,
        item_feature_size=item_feature_size,
    )
    
    batch_size = 2
    seq_len = 3
    user_ids = torch.tensor([0, 1])
    input_seq = torch.tensor([[1, 2, 3], [2, 3, 4]])
    input_seq_ts_bucket = torch.tensor([[1, 2, 3], [2, 3, 4]])
    item_features = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    target_item = torch.tensor([0, 1])
    
    output = model(user_ids, input_seq, input_seq_ts_bucket, item_features, target_item)
    
    assert output.shape == (batch_size, 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_ranker_recommend():
    num_users = 2
    num_items = 5
    embedding_dim = 16
    item_sequence_ts_bucket_size = 10
    bucket_embedding_dim = 8
    item_feature_size = 2
    
    model = Ranker(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        item_sequence_ts_bucket_size=item_sequence_ts_bucket_size,
        bucket_embedding_dim=bucket_embedding_dim,
        item_feature_size=item_feature_size,
    )
    
    users = torch.tensor([0, 1])
    item_sequences = torch.tensor([[1, 2, 3], [2, 3, 4]])
    item_ts_bucket_sequences = torch.tensor([[1, 2, 3], [2, 3, 4]])
    item_features = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
    item_indices = torch.tensor([0, 1, 2, 3, 4])
    k = 2
    
    topk_items = model.recommend(users, item_sequences, item_ts_bucket_sequences, item_features, item_indices, k)
    
    assert topk_items.shape == (num_users, k)
    assert torch.all(topk_items >= 0) and torch.all(topk_items < num_items)

# Test LitRanker
def test_lit_ranker_training_step(sample_df, sample_item_features):
    model = Ranker(
        num_users=3,
        num_items=5,
        embedding_dim=16,
        item_sequence_ts_bucket_size=10,
        bucket_embedding_dim=8,
        item_feature_size=2,
    )
    
    lit_ranker = LitRanker(
        model=model,
        learning_rate=0.001,
        l2_reg=1e-5,
        neg_to_pos_ratio=3,
    )
    
    dataset = UserItemBinaryDFDataset(
        df=sample_df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        timestamp_col="timestamp",
        item_feature=sample_item_features,
    )
    
    def collate_fn(batch):
        return {
            "user": torch.stack([x["user"] for x in batch]),
            "item": torch.stack([x["item"] for x in batch]),
            "item_sequence": torch.stack([x["item_sequence"] for x in batch]),
            "item_sequence_ts_bucket": torch.stack([x["item_sequence_ts_bucket"] for x in batch]),
            "item_feature": torch.stack([x["item_feature"] for x in batch]),
            "rating": torch.stack([x["rating"] for x in batch]),
        }
    
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    loss = lit_ranker.training_step(batch, batch_idx=0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_lit_ranker_validation_step(sample_df, sample_item_features):
    model = Ranker(
        num_users=3,
        num_items=5,
        embedding_dim=16,
        item_sequence_ts_bucket_size=10,
        bucket_embedding_dim=8,
        item_feature_size=2,
    )
    
    lit_ranker = LitRanker(
        model=model,
        learning_rate=0.001,
        l2_reg=1e-5,
        neg_to_pos_ratio=3,
    )
    
    dataset = UserItemBinaryDFDataset(
        df=sample_df,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        timestamp_col="timestamp",
        item_feature=sample_item_features,
    )
    
    def collate_fn(batch):
        return {
            "user": torch.stack([x["user"] for x in batch]),
            "item": torch.stack([x["item"] for x in batch]),
            "item_sequence": torch.stack([x["item_sequence"] for x in batch]),
            "item_sequence_ts_bucket": torch.stack([x["item_sequence_ts_bucket"] for x in batch]),
            "item_feature": torch.stack([x["item_feature"] for x in batch]),
            "rating": torch.stack([x["rating"] for x in batch]),
        }
    
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    loss = lit_ranker.validation_step(batch, batch_idx=0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert lit_ranker.val_roc_auc_metric.compute() >= 0


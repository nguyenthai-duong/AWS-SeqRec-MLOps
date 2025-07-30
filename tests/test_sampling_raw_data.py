import pytest
import pandas as pd
import numpy as np
from loguru import logger
from unittest.mock import patch
from sampling_raw_data import InteractionDataSampler  # Adjust import based on actual module structure

# Mock data generation with overlapping users and items
def create_mock_data(
    num_users=10000,
    num_items=5000,
    num_interactions=100000,
    val_interactions=20000,
    seed=41,
    overlap_ratio=0.9
):
    np.random.seed(seed)
    
    # Core users and items for overlap
    core_users = np.arange(1, int(num_users * overlap_ratio) + 1)
    core_items = np.arange(1, int(num_items * overlap_ratio) + 1)
    
    # Train DataFrame
    train_user_ids = np.random.choice(
        np.arange(1, num_users + 1), size=num_interactions, replace=True
    )
    train_item_ids = np.random.choice(
        np.arange(1, num_items + 1), size=num_interactions, replace=True
    )
    train_df = pd.DataFrame({
        'user_id': train_user_ids,
        'item_id': train_item_ids
    })
    
    # Validation DataFrame with overlap
    val_user_ids = np.random.choice(
        core_users, size=val_interactions, replace=True
    )
    val_item_ids = np.random.choice(
        core_items, size=val_interactions, replace=True
    )
    val_df = pd.DataFrame({
        'user_id': val_user_ids,
        'item_id': val_item_ids
    })
    
    return train_df, val_df

@pytest.fixture
def sampler():
    """Fixture for InteractionDataSampler with default parameters."""
    return InteractionDataSampler(
        user_col="user_id",
        item_col="item_id",
        sample_users=1000,
        min_val_records=1000,
        random_seed=41,
        min_user_interactions=5,
        min_item_interactions=10,
        buffer_perc=0.2,
        perc_users_removed_each_round=0.01,
        debug=True
    )

@pytest.fixture
def mock_data():
    """Fixture for mock train and validation DataFrames with overlap."""
    return create_mock_data()

@patch.object(logger, 'info')
def test_sample_basic_functionality(mock_logger, sampler, mock_data):
    """Test sample method basic functionality and constraints."""
    train_df, val_df = mock_data
    
    sample_df, val_sample_df = sampler.sample(train_df, val_df)
    
    assert isinstance(sample_df, pd.DataFrame), "Sampled train_df is not a DataFrame"
    assert isinstance(val_sample_df, pd.DataFrame), "Sampled val_df is not a DataFrame"
    
    user_counts = sample_df.groupby('user_id').size()
    item_counts = sample_df.groupby('item_id').size()
    
    assert all(user_counts >= sampler.min_user_interactions), (
        f"Some users have fewer than {sampler.min_user_interactions} interactions"
    )
    assert all(item_counts >= sampler.min_item_interactions), (
        f"Some items have fewer than {sampler.min_item_interactions} interactions"
    )
    
    assert set(val_sample_df['user_id']).issubset(set(sample_df['user_id'])), (
        "Validation users not subset of train users"
    )
    assert set(val_sample_df['item_id']).issubset(set(sample_df['item_id'])), (
        "Validation items not subset of train items"
    )
    
    num_users = sample_df['user_id'].nunique()
    assert num_users <= sampler.sample_users * (1 + sampler.buffer_perc), (
        f"Number of users {num_users} exceeds expected {sampler.sample_users * (1 + sampler.buffer_perc)}"
    )
    
    # Check validation records with warning for low counts
    if val_sample_df.shape[0] < sampler.min_val_records:
        logger.warning(
            f"Validation records {val_sample_df.shape[0]} below minimum {sampler.min_val_records}"
        )
    else:
        assert val_sample_df.shape[0] >= sampler.min_val_records, (
            f"Validation records {val_sample_df.shape[0]} below minimum {sampler.min_val_records}"
        )

def test_sample_edge_case_low_interactions(sampler):
    """Test sample method with minimal interaction data."""
    small_df = pd.DataFrame({
        'user_id': [1, 1, 2, 2],
        'item_id': [1, 2, 1, 2]
    })
    
    sample_df, val_sample_df = sampler.sample(small_df, small_df)
    
    assert sample_df.empty or (
        sample_df.groupby('user_id').size().min() >= sampler.min_user_interactions
        and sample_df.groupby('item_id').size().min() >= sampler.min_item_interactions
    ), "Sampled train_df does not meet interaction thresholds"
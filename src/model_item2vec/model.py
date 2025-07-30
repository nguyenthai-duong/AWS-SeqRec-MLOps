from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from model_item2vec.dataset import SkipGramDataset


class SkipGram(nn.Module):
    """SkipGram model for learning item embeddings using positive and negative sampling."""

    def __init__(self, num_items: int, embedding_dim: int):
        """Initialize the SkipGram model.

        Args:
            num_items (int): Total number of unique items in the vocabulary.
            embedding_dim (int): Dimensionality of the embedding vectors.
        """
        super().__init__()
        self.embeddings = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=num_items
        )
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, target_items: torch.Tensor, context_items: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between target and context items.

        Args:
            target_items (torch.Tensor): Tensor of target item indices with shape (batch_size,).
            context_items (torch.Tensor): Tensor of context item indices with shape (batch_size,).

        Returns:
            torch.Tensor: Predicted probabilities with shape (batch_size,).
        """
        target_embeds = self.embeddings(target_items)  # Shape: (batch_size, embedding_dim)
        context_embeds = self.embeddings(context_items)  # Shape: (batch_size, embedding_dim)

        # Compute dot product between target and context embeddings
        similarity_scores = torch.sum(target_embeds * context_embeds, dim=-1)  # Shape: (batch_size,)

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(similarity_scores)
        return probabilities

    def get_item_embedding(self, item_idx: int) -> torch.Tensor:
        """Retrieve the embedding vector for a specific item.

        Args:
            item_idx (int): Index of the item.

        Returns:
            torch.Tensor: Embedding vector for the specified item.
        """
        return self.embeddings(torch.tensor(item_idx, dtype=torch.long))

    def predict_train_batch(
        self,
        batch_input: Dict[str, Any],
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Predict scores for a batch of training data.

        Args:
            batch_input (Dict[str, Any]): Dictionary containing 'target_items' and 'context_items' tensors.
            device (torch.device, optional): Device to perform computations on. Defaults to CPU.

        Returns:
            torch.Tensor: Predicted scores for the batch.
        """
        target_items = batch_input["target_items"].to(device)
        context_items = batch_input["context_items"].to(device)
        return self.forward(target_items, context_items)

    @classmethod
    def get_expected_dataset_type(cls) -> List[type]:
        """Get the expected dataset type for training this model.

        Returns:
            List[type]: List containing the expected dataset type (SkipGramDataset).
        """
        return [SkipGramDataset]
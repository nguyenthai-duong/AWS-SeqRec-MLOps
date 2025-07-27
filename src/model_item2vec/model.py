from typing import Any, Dict

import torch
import torch.nn as nn
from dataset import SkipGramDataset


class SkipGram(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=num_items
        )
        nn.init.xavier_uniform_(
            self.embeddings.weight
        )  # Same performance with less total training time

    def forward(self, target_items, context_items):
        """
        :param target_items: Tensor of target items (batch_size,)
        :param context_items: Tensor of context items (batch_size,)
        """
        # Get the embeddings for the target and context items
        target_embeds = self.embeddings(target_items)  # (batch_size, embedding_dim)
        context_embeds = self.embeddings(context_items)  # (batch_size, embedding_dim)

        # Compute dot product between target and context embeddings
        similarity_scores = torch.sum(
            target_embeds * context_embeds, dim=-1
        )  # (batch_size,)

        # Apply sigmoid to get the probabilities
        probabilities = torch.sigmoid(similarity_scores)

        return probabilities

    def get_item_embedding(self, item_idx: int) -> torch.Tensor:
        """
        Get the embedding vector for a specific item.

        Args:
            item_idx (int): Index of the item

        Returns:
            torch.Tensor: Embedding vector of the item
        """
        return self.embeddings(torch.tensor(item_idx))

    def predict_train_batch(
        self, batch_input: Dict[str, Any], device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Predict scores for a batch of training data.

        Args:
            batch_input (Dict[str, Any]): A dictionary containing tensors with 'user' and 'item' indices.
            device (torch.device, optional): The device on which the model will run (CPU by default).

        Returns:
            torch.Tensor: The predicted scores for the batch.
        """
        target_items = batch_input["target_items"].to(device)
        context_items = batch_input["context_items"].to(device)
        return self.forward(target_items, context_items)

    @classmethod
    def get_expected_dataset_type(cls):
        """
        Returns the expected dataset type for training this model.
        """
        return [SkipGramDataset]

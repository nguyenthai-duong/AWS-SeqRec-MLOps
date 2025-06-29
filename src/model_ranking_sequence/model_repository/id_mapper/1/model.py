
import dill
import numpy as np
import logging
import triton_python_backend_utils as pb_utils
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def __init__(self):
        self.idm = None
        self.sequence_length = 10  # Match SEQ_LEN from script
        self.padding_value = -1   # Match padding_value from convert_asin_to_idx

    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(model_dir, "id_mapper.pkl")
        logger.info(f"Loading IDMapper from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.idm = dill.load(f)
        logger.info("IDMapper initialized")

    def convert_to_idx(self, sequence, sequence_length, padding_value):
        # Handle None or empty sequences
        if sequence is None or len(sequence) == 0:
            return np.array([padding_value] * sequence_length, dtype=np.int64)

        # Convert items to indices
        indices = []
        for item in sequence:
            if item == -1:  # Preserve padding_value
                indices.append(padding_value)
            else:
                try:
                    idx = self.idm.get_item_index(str(item))
                    indices.append(idx)
                except (KeyError, ValueError):
                    indices.append(padding_value)  # Handle invalid items

        # Pad sequence if needed
        padding_needed = sequence_length - len(indices)
        if padding_needed > 0:
            indices = [padding_value] * padding_needed + indices
        elif padding_needed < 0:
            indices = indices[-sequence_length:]  # Truncate if too long

        return np.array(indices, dtype=np.int64)

    def execute(self, requests):
        responses = []
        for request in requests:
            user_ids = pb_utils.get_input_tensor_by_name(request, "user_ids").as_numpy()  # Shape: [batch_size, -1]
            target_items = pb_utils.get_input_tensor_by_name(request, "target_items").as_numpy()  # Shape: [batch_size, 1]
            input_seq = pb_utils.get_input_tensor_by_name(request, "input_seq").as_numpy()  # Shape: [batch_size, 10]

            batch_size = user_ids.shape[0]
            logger.info("Batch size: %d", batch_size)

            # Process user_ids: Take the first non-empty string per batch item
            user_indices = np.array([
                self.idm.get_user_index(uid[0].decode('utf-8')) for uid in user_ids
            ], dtype=np.int64).reshape(batch_size, 1)  # Shape: [batch_size, 1]

            # Process target_items
            target_indices = np.array([
                self.idm.get_item_index(item[0].decode('utf-8')) for item in target_items
            ], dtype=np.int64).reshape(batch_size, 1)  # Shape: [batch_size, 1]

            # Process input_seq with padding
            seq_indices = np.array([
                self.convert_to_idx(seq, self.sequence_length, self.padding_value)
                for seq in input_seq
            ], dtype=np.int64)  # Shape: [batch_size, 10]

            logger.info("user_indices shape: %s", user_indices.shape)
            logger.info("seq_indices shape: %s", seq_indices.shape)
            logger.info("target_indices shape: %s", target_indices.shape)

            user_indices_tensor = pb_utils.Tensor("user_indices", user_indices)
            seq_indices_tensor = pb_utils.Tensor("seq_indices", seq_indices)
            target_indices_tensor = pb_utils.Tensor("target_indices", target_indices)

            response = pb_utils.InferenceResponse(output_tensors=[
                user_indices_tensor,
                seq_indices_tensor,
                target_indices_tensor
            ])
            responses.append(response)
        return responses

    def finalize(self):
        logger.info("IDMapper finalized")

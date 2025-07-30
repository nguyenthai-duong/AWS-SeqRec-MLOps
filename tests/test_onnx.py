"""
Test script for Triton Inference Server ensemble model (single-item test)
"""
import numpy as np
import logging
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Triton configuration
TRITON_URL = "localhost:8001"
MODEL_NAME = "ensemble"
SEQ_LEN = 10
MAX_ITEM_ID_LEN = 1


def create_test_inputs():
    """Create test inputs for a single item based on TEST_DATA."""
    # Use data from the first entry in TEST_DATA
    user_id = "AFI4TKPAEMA6VBRHQ25MUXLHEIBA"
    input_seq = [
    "B09PH8Z5R8",
    "B00HQ0WAJG",
    "B00NHQGE04",
    "B000GHVHRW",
    "B002N4KK7O",
    "B00B4KNZH0",
    "B00146G71U",
    "B08RXLPBT5"
]
    input_seq_ts_bucket = [-1, -1, 8, 7, 7, 6, 6, 6, 0, 0]
    item = "B07N29HQMN"
    categories = ["Toys & Games", "Games & Accessories", "Board Games"]
    price = 13.64
    main_category = "Toys & Games"
    rating_cnt_365d = 13
    rating_avg_365d = 8.0
    rating_cnt_90d = 0
    rating_avg_90d = 0.0
    rating_cnt_30d = 0
    rating_avg_30d = 0.0
    rating_cnt_7d = 0
    rating_avg_7d = 0.0

    # String inputs
    user_ids = np.array([user_id], dtype=object).reshape(1, 1)  # Shape: [1, 1]

    items = np.array([item], dtype=object).reshape(1, MAX_ITEM_ID_LEN)  # Shape: [1, 1]

    categories = np.array(["__".join(categories)], dtype=object).reshape(1, 1)
    main_categories = np.array([main_category], dtype=object).reshape(1, 1)

    # Numeric inputs (single value per item)
    prices = np.array([price], dtype=np.float32).reshape(1, 1)  # Shape: [1, 1]
    rating_cnt_365d_array = np.array([rating_cnt_365d], dtype=np.int64).reshape( 1, 1)  # Shape: [1, 1]
    rating_avg_365d_array = np.array([rating_avg_365d], dtype=np.float32).reshape(1, 1)  # Shape: [1, 1]
    rating_cnt_90d_array = np.array([rating_cnt_90d], dtype=np.int64).reshape(1, 1)  # Shape: [1, 1]
    rating_avg_90d_array = np.array([rating_avg_90d], dtype=np.float32).reshape(1, 1)  # Shape: [1, 1]
    rating_cnt_30d_array = np.array([rating_cnt_30d], dtype=np.int64).reshape(1, 1)  # Shape: [1, 1]
    rating_avg_30d_array = np.array([rating_avg_30d], dtype=np.float32).reshape(1, 1)  # Shape: [1, 1]
    rating_cnt_7d_array = np.array([rating_cnt_7d], dtype=np.int64).reshape(1, 1)  # Shape: [1, 1]
    rating_avg_7d_array = np.array([rating_avg_7d], dtype=np.float32).reshape(1, 1)  # Shape: [1, 1]

    # Prepare inputs
    inputs = {
        "user_ids": user_ids,  # Shape: [1, 32]
        "input_seq": np.array([input_seq], dtype=object),  # Shape: [1, 10]
        "input_seq_ts_bucket": np.array([input_seq_ts_bucket], dtype=np.int64),  # Shape: [1, 10]
        "items": items,  # Shape: [1, 1]
        "categories": categories,  # Shape: [1, 100]
        "prices": prices,  # Shape: [1, 1]
        "main_categories": main_categories,  # Shape: [1, 100]
        "rating_cnt_365d": rating_cnt_365d_array,  # Shape: [1, 1]
        "rating_avg_365d": rating_avg_365d_array,  # Shape: [1, 1]
        "rating_cnt_90d": rating_cnt_90d_array,  # Shape: [1, 1]
        "rating_avg_90d": rating_avg_90d_array,  # Shape: [1, 1]
        "rating_cnt_30d": rating_cnt_30d_array,  # Shape: [1, 1]
        "rating_avg_30d": rating_avg_30d_array,  # Shape: [1, 1]
        "rating_cnt_7d": rating_cnt_7d_array,  # Shape: [1, 1]
        "rating_avg_7d": rating_avg_7d_array,  # Shape: [1, 1]
    }

    return inputs

def main():
    # Create Triton client
    try:
        client = InferenceServerClient(url=TRITON_URL)
        logger.info("Connected to Triton server at %s", TRITON_URL)
    except Exception as e:
        logger.error("Failed to connect to Triton server: %s", e)
        raise

    # Check server and model status
    try:
        if not client.is_server_live():
            logger.error("Triton server is not live")
            raise RuntimeError("Triton server is not live")
        if not client.is_model_ready(MODEL_NAME):
            logger.error("Model %s is not ready", MODEL_NAME)
            raise RuntimeError(f"Model {MODEL_NAME} is not ready")
        logger.info("Server live and model %s ready", MODEL_NAME)
    except Exception as e:
        logger.error("Error checking server/model status: %s", e)
        raise

    # Prepare inputs
    input_data = create_test_inputs()

    # Create InferInput objects
    inputs = []
    try:
        for name, data in input_data.items():
            if data.dtype == object:
                data_type = "BYTES"
            elif data.dtype == np.int64:
                data_type = "INT64"
            elif data.dtype == np.float32:
                data_type = "FP32"
            else:
                raise ValueError(f"Unsupported dtype {data.dtype} for input {name}")

            infer_input = InferInput(name, data.shape, data_type)
            infer_input.set_data_from_numpy(data)
            inputs.append(infer_input)
            logger.debug("Input %s: shape=%s, dtype=%s", name, data.shape, data_type)
    except Exception as e:
        logger.error("Error creating inputs: %s", e)
        raise

    # Define output
    outputs = [InferRequestedOutput("output")]

    # Perform inference
    try:
        response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        output = response.as_numpy("output")
        logger.info("Inference Output: %s", output)
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise

if __name__ == "__main__":
    main()
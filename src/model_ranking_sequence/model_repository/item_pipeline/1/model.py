
import dill
import numpy as np
import pandas as pd
import logging
import triton_python_backend_utils as pb_utils
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def __init__(self):
        self.pipeline = None
    
    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(model_dir, "item_metadata_pipeline.pkl")
        logger.info(f"Loading item pipeline from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.pipeline = dill.load(f)
        logger.info("Item pipeline initialized")
    
    def execute(self, requests):
        responses = []
        for request in requests:
            items = pb_utils.get_input_tensor_by_name(request, "items").as_numpy()
            categories = pb_utils.get_input_tensor_by_name(request, "categories").as_numpy()
            prices = pb_utils.get_input_tensor_by_name(request, "prices").as_numpy()
            main_categories = pb_utils.get_input_tensor_by_name(request, "main_categories").as_numpy()
            rating_cnt_365d = pb_utils.get_input_tensor_by_name(request, "rating_cnt_365d").as_numpy()
            rating_avg_365d = pb_utils.get_input_tensor_by_name(request, "rating_avg_365d").as_numpy()
            rating_cnt_90d = pb_utils.get_input_tensor_by_name(request, "rating_cnt_90d").as_numpy()
            rating_avg_90d = pb_utils.get_input_tensor_by_name(request, "rating_avg_90d").as_numpy()
            rating_cnt_30d = pb_utils.get_input_tensor_by_name(request, "rating_cnt_30d").as_numpy()
            rating_avg_30d = pb_utils.get_input_tensor_by_name(request, "rating_avg_30d").as_numpy()
            rating_cnt_7d = pb_utils.get_input_tensor_by_name(request, "rating_cnt_7d").as_numpy()
            rating_avg_7d = pb_utils.get_input_tensor_by_name(request, "rating_avg_7d").as_numpy()
            
            batch_size = items.shape[0]
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"items shape: {items.shape}")
            logger.info(f"categories shape: {categories.shape}")
            logger.info(f"prices shape: {prices.shape}")
            
            df_data = {
                "parent_asin": [item.decode('utf-8') for item in items.reshape(batch_size, -1)[:, 0]],
                "categories": [cat.decode('utf-8').replace('__', '|') for cat in categories.reshape(batch_size, -1)[:, 0]],
                "price": prices.reshape(batch_size, -1)[:, 0].astype(np.float32),
                "main_category": [mc.decode('utf-8') for mc in main_categories.reshape(batch_size, -1)[:, 0]],
                "parent_asin_rating_cnt_365d": rating_cnt_365d.reshape(batch_size, -1)[:, 0].astype(np.int64),
                "parent_asin_rating_avg_prev_rating_365d": rating_avg_365d.reshape(batch_size, -1)[:, 0].astype(np.float32),
                "parent_asin_rating_cnt_90d": rating_cnt_90d.reshape(batch_size, -1)[:, 0].astype(np.int64),
                "parent_asin_rating_avg_prev_rating_90d": rating_avg_90d.reshape(batch_size, -1)[:, 0].astype(np.float32),
                "parent_asin_rating_cnt_30d": rating_cnt_30d.reshape(batch_size, -1)[:, 0].astype(np.int64),
                "parent_asin_rating_avg_prev_rating_30d": rating_avg_30d.reshape(batch_size, -1)[:, 0].astype(np.float32),
                "parent_asin_rating_cnt_7d": rating_cnt_7d.reshape(batch_size, -1)[:, 0].astype(np.int64),
                "parent_asin_rating_avg_prev_rating_7d": rating_avg_7d.reshape(batch_size, -1)[:, 0].astype(np.float32)
            }
            
            for key, value in df_data.items():
                logger.info(f"DataFrame column {key} length: {len(value)}")
            
            df = pd.DataFrame(df_data)
            df.fillna(0.0, inplace=True)
            features = self.pipeline.transform(df).astype(np.float32)
            logger.info(f"item_features shape: {features.shape}")
            
            features_tensor = pb_utils.Tensor("item_features", features)
            response = pb_utils.InferenceResponse(output_tensors=[features_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        logger.info("Item pipeline finalized")

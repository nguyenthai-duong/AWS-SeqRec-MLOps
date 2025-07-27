import os
import time

import papermill as pm
from loguru import logger

run_timestamp = int(time.time())
output_dir = f"output/{run_timestamp}"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"{run_timestamp=}")
logger.info(f"Notebook outputs will be saved to {output_dir}")

pm.execute_notebook(
    "001-features.ipynb", f"{output_dir}/001-features.ipynb", kernel_name="python3"
)

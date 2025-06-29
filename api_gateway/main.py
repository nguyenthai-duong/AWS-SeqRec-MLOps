import logging
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import redis
import json
import numpy as np
from tritonclient.grpc.aio import InferenceServerClient, InferInput, InferRequestedOutput
import os
import time
import grpc

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Per-worker Triton client via FastAPI dependency
def get_triton_client():
    if not hasattr(app.state, "triton_client"):
        app.state.triton_client = InferenceServerClient(url=TRITON_URL, ssl=False)
    return app.state.triton_client

# Global HTTP session
http_session: aiohttp.ClientSession = None

# Redis client setup with connection pool
redis_pool = redis.ConnectionPool(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    db=int(os.environ.get("REDIS_DB", 0)),
    password=os.environ.get("REDIS_PASSWORD", "123456"),
    decode_responses=True
)
redis_client = redis.Redis(connection_pool=redis_pool)

# Config for external services
USER_FEATURE_URL = os.environ.get("USER_FEATURE_URL", "http://localhost:8005/user_features")
ITEM_FEATURE_URL = os.environ.get("ITEM_FEATURE_URL", "http://localhost:8005/features_parent_asin")
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8001")
MODEL_NAME = os.environ.get("MODEL_NAME", "ensemble")
SEQ_LEN = int(os.environ.get("SEQ_LEN", 10))
MAX_ITEM_ID_LEN = int(os.environ.get("MAX_ITEM_ID_LEN", 1))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))  # Default, can override with env
MAX_SEMAPHORE = int(os.environ.get("MAX_FETCH_CONCURRENCY", 50))
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", 100))

# Request models
class UserRequest(BaseModel):
    user_id: str

class ItemRequest(BaseModel):
    parent_asin: str

def safe_float(value, default=0.0):
    if value is None or value == "None":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    if value is None or value == "None":
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

@app.on_event("startup")
async def startup_event():
    global http_session
    http_session = aiohttp.ClientSession()
    logger.info("Initialized aiohttp ClientSession")
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except redis.RedisError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise RuntimeError("Redis connection failed")

@app.on_event("shutdown")
async def shutdown_event():
    global http_session
    if http_session:
        await http_session.close()
        logger.info("Closed aiohttp ClientSession")
    # Triton client cleanup per worker (if exists)
    if hasattr(app.state, "triton_client"):
        await app.state.triton_client.close()
        del app.state.triton_client

# 1. Popular items
@app.get("/popular_items")
def get_popular_items(top_k: int = 10):
    try:
        redis_key = "popular_parent_asin_score"
        top_items = redis_client.zrevrange(redis_key, 0, top_k - 1, withscores=True)
        result = [
            {"rank": i + 1, "parent_asin": asin, "score": score}
            for i, (asin, score) in enumerate(top_items)
        ]
        return {"popular_items": result}
    except redis.RedisError as e:
        logger.error(f"Redis error in get_popular_items: {e}")
        raise HTTPException(status_code=500, detail="Redis error")

# 2. Get user features (proxy)
@app.get("/user_features")
async def get_user_features(user_id: str = Query("AFI4TKPAEMA6VBRHQ25MUXLHEIBA")):
    try:
        async with http_session.post(USER_FEATURE_URL, json={"user_id": user_id}, timeout=3) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"Error calling user service: {e}")
        raise HTTPException(status_code=502, detail=f"Error calling user service: {e}")

# 3. Get item features (proxy)
@app.get("/features_parent_asin")
async def get_item_features(parent_asin: str = Query("B000N178E2")):
    try:
        async with http_session.post(ITEM_FEATURE_URL, json={"parent_asin": parent_asin}, timeout=3) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"Error calling item service: {e}")
        raise HTTPException(status_code=502, detail=f"Error calling item service: {e}")

# 4. Similar items from Redis
@app.get("/similar_items")
def get_similar_items(item_id: str = Query("B000N178E2")):
    try:
        key = f"rec:{item_id}"
        rec_json = redis_client.get(key)
        if rec_json:
            rec_data = json.loads(rec_json)
            return {"item_id": item_id, "recommendations": rec_data}
        else:
            raise HTTPException(status_code=404, detail=f"No recommendation found for item {item_id}")
    except redis.RedisError as e:
        logger.error(f"Redis error in get_similar_items: {e}")
        raise HTTPException(status_code=500, detail="Redis error")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for similar items {key}: {e}")
        raise HTTPException(status_code=500, detail="Invalid recommendation data")

# 5. Infer scores with batching and per-worker triton client
@app.get("/infer")
async def infer_score(user_id: str = Query("AFI4TKPAEMA6VBRHQ25MUXLHEIBA")):
    total_start = time.time()
    try:
        ### 1. Fetch user features
        t0 = time.time()
        logger.debug(f"Fetching user features for user_id: {user_id}")
        async with http_session.post(USER_FEATURE_URL, json={"user_id": user_id}, timeout=3) as response:
            response.raise_for_status()
            try:
                user_features = await response.json()
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for user features: {e}")
                raise HTTPException(status_code=502, detail="Invalid user features response")
        t1 = time.time()
        logger.info(f"[TIMING] Fetch user_features: {t1 - t0:.3f}s")

        asin_string = user_features.get("user_rating_list_10_recent_asin", [""])[0]
        input_seq = asin_string.split(",") if asin_string else []
        input_seq_ts = user_features.get("item_sequence_ts_bucket", [[]])[0]

        ### 2. Fetch popular items
        t2 = time.time()
        try:
            popular_items = redis_client.zrevrange("popular_parent_asin_score", 0, 19)
        except redis.RedisError as e:
            logger.error(f"Redis error fetching popular items: {e}")
            raise HTTPException(status_code=500, detail="Redis error")
        t3 = time.time()
        logger.info(f"[TIMING] Fetch popular_items (Redis): {t3 - t2:.3f}s")

        ### 3. Fetch similar items
        last_item = input_seq[-1] if input_seq else None
        t4 = time.time()
        similar_items = []
        if last_item:
            key = f"rec:{last_item}"
            try:
                rec_json = redis_client.get(key)
                if rec_json:
                    try:
                        similar_items = json.loads(rec_json).get("rec_items", [])
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error for similar items {key}: {e}")
            except redis.RedisError as e:
                logger.error(f"Redis error fetching similar items {key}: {e}")
        t5 = time.time()
        logger.info(f"[TIMING] Fetch similar_items (Redis): {t5 - t4:.3f}s")

        # Filter out seen items and limit candidates
        seen_set = set(input_seq)
        candidates = list({*popular_items, *similar_items} - seen_set)[:MAX_CANDIDATES]
        if not candidates:
            logger.warning("No valid candidates found")
            raise HTTPException(status_code=404, detail="No valid candidates")

        # Prepare user-specific inputs
        user_ids = np.array([user_id], dtype=object).reshape(1, 1)
        padded_seq = input_seq[-SEQ_LEN:] + ["-1"] * max(0, SEQ_LEN - len(input_seq))
        padded_seq_ts = input_seq_ts[-SEQ_LEN:] + [-1] * max(0, SEQ_LEN - len(input_seq_ts))
        input_seq_array = np.array([padded_seq], dtype=object)
        input_seq_ts_array = np.array([padded_seq_ts], dtype=np.int64)

        ### 4. Fetch item features concurrently with caching and semaphore
        t6 = time.time()
        async def fetch_item_features(item, semaphore):
            cache_key = f"item_features:{item}"
            cached = redis_client.get(cache_key)
            if cached:
                try:
                    return item, json.loads(cached)
                except json.JSONDecodeError:
                    logger.error(f"Invalid cached data for {item}")

            timeout = ClientTimeout(total=2, connect=0.5, sock_read=1.5)
            async with semaphore:
                for attempt in range(3):
                    try:
                        async with http_session.post(ITEM_FEATURE_URL, json={"parent_asin": item}, timeout=timeout) as response:
                            response.raise_for_status()
                            features = await response.json()
                            try:
                                redis_client.setex(cache_key, 300, json.dumps(features))  # Cache 5 min
                            except redis.RedisError as e:
                                logger.error(f"Failed to cache item features for {item}: {e}")
                            return item, features
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.warning(f"Attempt {attempt + 1} failed for {item}: {str(e)}")
                        if attempt == 2:
                            logger.error(f"Failed to fetch item features for {item}: {str(e)}")
                            return item, None
                        await asyncio.sleep(0.1 * (attempt + 1))

        semaphore = asyncio.Semaphore(MAX_SEMAPHORE)
        item_features_tasks = [fetch_item_features(item, semaphore) for item in candidates]
        item_features_results = await asyncio.gather(*item_features_tasks, return_exceptions=True)
        t7 = time.time()
        logger.info(f"[TIMING] Fetch {len(candidates)} item_features (async): {t7 - t6:.3f}s")

        valid_item_features = [result for result in item_features_results if isinstance(result, tuple) and result[1] is not None]
        if not valid_item_features:
            logger.warning("No valid item features found")
            raise HTTPException(status_code=404, detail="No valid item features found")

        ### 5. Triton inference with dynamic batching (auto-chunk by BATCH_SIZE)
        t8 = time.time()
        triton_client = get_triton_client()
        async def infer_batch(batch_items, batch_features):
            batch_size = len(batch_items)
            # Prepare inputs
            items = np.array(batch_items, dtype=object).reshape(batch_size, MAX_ITEM_ID_LEN)
            categories = np.array(["__".join(f.get("categories", [[]])[0]) for f in batch_features], dtype=object).reshape(batch_size, 1)
            main_categories = np.array([f.get("main_category", [""])[0] for f in batch_features], dtype=object).reshape(batch_size, 1)
            prices = np.array([safe_float(f.get("price", [0])[0]) for f in batch_features], dtype=np.float32).reshape(batch_size, 1)
            rating_cnt_365d = np.array([safe_int(f.get("parent_asin_rating_cnt_365d", [0])[0]) for f in batch_features], dtype=np.int64).reshape(batch_size, 1)
            rating_avg_365d = np.array([safe_float(f.get("parent_asin_rating_avg_prev_rating_365d", [0])[0]) for f in batch_features], dtype=np.float32).reshape(batch_size, 1)
            rating_cnt_90d = np.array([safe_int(f.get("parent_asin_rating_cnt_90d", [0])[0]) for f in batch_features], dtype=np.int64).reshape(batch_size, 1)
            rating_avg_90d = np.array([safe_float(f.get("parent_asin_rating_avg_prev_rating_90d", [0])[0]) for f in batch_features], dtype=np.float32).reshape(batch_size, 1)
            rating_cnt_30d = np.array([safe_int(f.get("parent_asin_rating_cnt_30d", [0])[0]) for f in batch_features], dtype=np.int64).reshape(batch_size, 1)
            rating_avg_30d = np.array([safe_float(f.get("parent_asin_rating_avg_prev_rating_30d", [0])[0]) for f in batch_features], dtype=np.float32).reshape(batch_size, 1)
            rating_cnt_7d = np.array([safe_int(f.get("parent_asin_rating_cnt_7d", [0])[0]) for f in batch_features], dtype=np.int64).reshape(batch_size, 1)
            rating_avg_7d = np.array([safe_float(f.get("parent_asin_rating_avg_prev_rating_7d", [0])[0]) for f in batch_features], dtype=np.float32).reshape(batch_size, 1)

            # Repeat user sequence for batch
            user_ids = np.repeat(user_ids_infer, batch_size, axis=0)
            input_seq = np.repeat(input_seq_array, batch_size, axis=0)
            input_seq_ts_bucket = np.repeat(input_seq_ts_array, batch_size, axis=0)

            input_data = {
                "user_ids": user_ids,
                "input_seq": input_seq,
                "input_seq_ts_bucket": input_seq_ts_bucket,
                "items": items,
                "categories": categories,
                "main_categories": main_categories,
                "prices": prices,
                "rating_cnt_365d": rating_cnt_365d,
                "rating_avg_365d": rating_avg_365d,
                "rating_cnt_90d": rating_cnt_90d,
                "rating_avg_90d": rating_avg_90d,
                "rating_cnt_30d": rating_cnt_30d,
                "rating_avg_30d": rating_avg_30d,
                "rating_cnt_7d": rating_cnt_7d,
                "rating_avg_7d": rating_avg_7d,
            }

            inputs = []
            for name, data in input_data.items():
                dtype = data.dtype
                if dtype == object:
                    dtype_str = "BYTES"
                elif dtype == np.int64:
                    dtype_str = "INT64"
                elif dtype == np.float32:
                    dtype_str = "FP32"
                else:
                    raise ValueError(f"Unsupported dtype {dtype} for {name}")
                inp = InferInput(name, data.shape, dtype_str)
                inp.set_data_from_numpy(data)
                inputs.append(inp)

            outputs = [InferRequestedOutput("output")]
            try:
                response = await triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
                scores = response.as_numpy("output").flatten()
                return [{"item_id": item, "score": float(score)} for item, score in zip(batch_items, scores)]
            except Exception as e:
                logger.error(f"Triton inference failed for batch: {str(e)}")
                return []

        # Prepare static arrays (reuse for all batch)
        user_ids_infer = np.array([user_id], dtype=object).reshape(1, 1)
        input_seq_array = np.array([padded_seq], dtype=object)
        input_seq_ts_array = np.array([padded_seq_ts], dtype=np.int64)

        inference_results = []
        for i in range(0, len(valid_item_features), BATCH_SIZE):
            batch = valid_item_features[i:i + BATCH_SIZE]
            batch_items = [item for item, _ in batch]
            batch_features = [features for _, features in batch]
            results = await infer_batch(batch_items, batch_features)
            inference_results.extend(results)
        t9 = time.time()
        logger.info(f"[TIMING] Triton inference {len(valid_item_features)} items (batched): {t9 - t8:.3f}s")

        if not inference_results:
            logger.warning("No valid scores computed")
            raise HTTPException(status_code=404, detail="No valid scores computed")
        inference_results.sort(key=lambda x: x["score"], reverse=True)
        total_end = time.time()
        logger.info(f"[TIMING] Total /infer request: {total_end - total_start:.3f}s")
        return {"user_id": user_id, "recommendations": inference_results[:10]}

    except Exception as e:
        logger.error(f"Error in /infer: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


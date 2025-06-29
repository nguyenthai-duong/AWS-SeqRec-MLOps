import gradio as gr
import pandas as pd
import requests
import logging
import numpy as np
from typing import Dict, List
from feast import FeatureStore
from datetime import datetime, timedelta
import time

# —— Logging setup ——
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# —— Load metadata ——
try:
    meta_df = pd.read_parquet('../data/raw_meta.parquet')
    logger.info("Loaded raw_meta.parquet successfully")
except Exception as e:
    logger.error(f"Failed to load parquet file: {e}")
    raise

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="../.env")
    store = FeatureStore(repo_path=".")
    logger.info("Initialized Feast Feature Store")
except Exception as e:
    logger.error(f"Failed to initialize Feature Store: {e}")
    raise

# —— Mock users & API ——
user_list = [
    "AFI4TKPAEMA6VBRHQ25MUXLHEIBA",
    "AGBOFSSHGILKH73MJZUUOTRCD4CA",
    "AGADWSNI4D65TO2LXO2PYRFWGB3A",
    "AFO2VP5GWU5PF44UFVG5HTOHYLHA",
    "AHURNOZ3IY2UBHUDAM2D22XTSVFA"
]
REC_API_URL = "http://localhost:8009/infer"
PURCHASE_API_URL = "http://localhost:8005/user_features"
feedback_store: Dict[str, List[str]] = {}
product_html_cache: Dict[str, Dict[str, str]] = {}

# —— Helper functions for timestamp bucketing ——
def bucketize_seconds_diff(seconds):
    if seconds < 60 * 10: return 0
    if seconds < 60 * 60: return 1
    if seconds < 60 * 60 * 24: return 2
    if seconds < 60 * 60 * 24 * 7: return 3
    if seconds < 60 * 60 * 24 * 30: return 4
    if seconds < 60 * 60 * 24 * 365: return 5
    if seconds < 60 * 60 * 24 * 365 * 3: return 6
    if seconds < 60 * 60 * 24 * 365 * 5: return 7
    if seconds < 60 * 60 * 24 * 365 * 10: return 8
    return 9

def from_ts_to_bucket(ts, current_ts):
    if current_ts is None:
        logger.warning("current_ts is None, using current time")
        current_ts = int(time.time())
    seconds_diff = current_ts - ts
    logger.debug(f"Calculating bucket: ts={ts}, current_ts={current_ts}, seconds_diff={seconds_diff}")
    return bucketize_seconds_diff(seconds_diff)

def pad_timestamp_sequence(inp, sequence_length=10, padding_value=-1):
    if inp is None or inp.strip() == "":
        logger.warning("Empty timestamp input, returning padded list")
        return [padding_value] * sequence_length
    try:
        timestamps = [x.strip() for x in inp.split(",") if x.strip()]
        inp_list = []
        for ts in timestamps:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
                unix_ts = int(dt.timestamp())
                inp_list.append(unix_ts)
            except ValueError:
                try:
                    dt = datetime.strptime(ts[:26] + 'Z', "%Y-%m-%dT%H:%M:%S.%fZ")
                    unix_ts = int(dt.timestamp())
                    inp_list.append(unix_ts)
                except ValueError:
                    try:
                        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                        unix_ts = int(dt.timestamp())
                        inp_list.append(unix_ts)
                    except ValueError:
                        logger.warning(f"Invalid timestamp format: {ts}")
                        continue
        padding_needed = sequence_length - len(inp_list)
        if padding_needed > 0:
            inp_list = [padding_needed] * padding_needed + inp_list
        return inp_list[:sequence_length]
    except Exception as e:
        logger.error(f"Error in pad_timestamp_sequence: {e}")
        return [padding_value] * sequence_length

def calc_sequence_timestamp_bucket(timestamps, current_ts):
    output = []
    for x in timestamps:
        x_i = int(x)
        if x_i == -1:
            output.append(x_i)
        else:
            bucket = from_ts_to_bucket(x_i, current_ts)
            output.append(bucket)
    return output

# —— Helpers ——
def get_recommendations(user_id: str) -> List[Dict]:
    try:
        r = requests.get(REC_API_URL, params={"user_id": user_id}, timeout=5)
        r.raise_for_status()
        data = r.json()
        logger.info(f"Fetched recommendations for {user_id}")
        return data.get("recommendations", [])[:10]
    except Exception as e:
        logger.error(f"Error fetching recs: {e}")
        return []

def get_purchased_products(user_id: str) -> List[Dict]:
    try:
        payload = {"user_id": user_id}
        r = requests.post(PURCHASE_API_URL, json=payload, headers={'Content-Type': 'application/json'}, timeout=5)
        r.raise_for_status()
        data = r.json()
        asins = data.get("user_rating_list_10_recent_asin", [""])[0].split(",")
        timestamps = data.get("user_rating_list_10_recent_asin_timestamp", [""])[0].split(",")
        paired = [(asin, ts) for asin, ts in zip(asins, timestamps) if asin and ts]
        paired.sort(key=lambda x: datetime.strptime(x[1], "%Y-%m-%dT%H:%M:%S.%fZ") if x[1] else datetime.min, reverse=True)
        logger.info(f"Fetched and sorted purchased products for {user_id}")
        return [{"asin": asin, "timestamp": ts} for asin, ts in paired[:5]]
    except Exception as e:
        logger.error(f"Error fetching purchased products: {e}")
        return []

def get_product_details(asin: str) -> Dict:
    try:
        prod = meta_df[meta_df['parent_asin'] == asin].iloc[0].to_dict()
        imgs = prod.get('images', {})
        url = None
        for key in ('large', 'hi_res'):
            arr = imgs.get(key)
            if isinstance(arr, (list, np.ndarray)) and len(arr) > 0:
                url = arr[0]
                break
        if not url:
            url = 'https://via.placeholder.com/300x300?text=No+Image'
        cats = prod.get('categories', [])
        if isinstance(cats, np.ndarray):
            cats = cats.tolist()
        if not isinstance(cats, list):
            cats = [cats] if cats else []
        if not cats:
            cats = [prod.get('main_category', 'Unknown')]
        cats = [str(c) for c in cats if c]
        return {
            "title": prod.get('title', 'No title'),
            "image_url": url,
            "rating": prod.get('average_rating', 0.0),
            "reviews": prod.get('rating_number', 0),
            "price": prod.get('price', '?'),
            "store": prod.get('store', '?'),
            "categories": cats,
            "asin": asin
        }
    except Exception:
        logger.warning(f"No product for {asin}")
        return {
            "title": "Unknown",
            "image_url": "https://via.placeholder.com/300x300?text=No+Image",
            "rating": 0.0, "reviews": 0,
            "price": "?", "store": "?", "categories": ["Unknown"],
            "asin": asin
        }

def toggle_like(user_id: str, asin: str, liked: List[bool], idx: int, buttons: List[Dict]):
    if user_id not in feedback_store:
        feedback_store[user_id] = []
    if asin in feedback_store[user_id]:
        feedback_store[user_id].remove(asin)
        liked[idx] = False
        buttons[idx] = {"value": '<i class="far fa-heart"></i> Like', "variant": "secondary", "index": idx}
        msg = f"Removed like: {asin}"
    else:
        feedback_store[user_id].append(asin)
        liked[idx] = True
        buttons[idx] = {"value": '<i class="fas fa-heart"></i> Liked', "variant": "primary", "index": idx}
        msg = f"Liked: {asin}"
    product_html = product_html_cache.get(user_id, {}).get('recs', [])
    if idx < len(product_html):
        det = get_product_details(asin)
        product_html[idx] = render_single_rec(det, idx, liked[idx], user_id, recs=get_recommendations(user_id))
        product_html_cache[user_id]['recs'] = product_html
    logger.debug(f"Current feedback store for {user_id}: {feedback_store[user_id]}")
    return msg, liked, buttons, "".join(product_html_cache[user_id]['recs']), "".join(product_html_cache[user_id]['purchased'])

def submit_likes(user_id: str):
    items = feedback_store.get(user_id, [])
    if not items:
        return "No items liked.", "", [], [], [], ""
    print(f"\nLiked items for user {user_id}:")
    for i, asin in enumerate(items, 1):
        product = get_product_details(asin)
        print(f"{i}. ASIN: {asin} - Title: {product['title'][:50]}...")
    try:
        payload = {"user_id": user_id}
        r = requests.post(PURCHASE_API_URL, json=payload, headers={'Content-Type': 'application/json'}, timeout=5)
        r.raise_for_status()
        data = r.json()
        recent_asins = data.get("user_rating_list_10_recent_asin", [""])[0].split(",")
        recent_timestamps = data.get("user_rating_list_10_recent_asin_timestamp", [""])[0].split(",")
        logger.info(f"Fetched purchase history for {user_id}")
    except Exception as e:
        logger.error(f"Error fetching purchase history: {e}")
        recent_asins = []
        recent_timestamps = []
    paired_history = [(asin, ts) for asin, ts in zip(recent_asins, recent_timestamps) if asin and ts]
    if not paired_history:
        logger.warning(f"No valid purchase history for {user_id}")
        paired_history = []
    if recent_timestamps:
        try:
            parsed_timestamps = []
            for ts in recent_timestamps:
                if ts:
                    try:
                        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
                        parsed_timestamps.append(dt)
                    except ValueError:
                        try:
                            dt = datetime.strptime(ts[:26] + 'Z', "%Y-%m-%dT%H:%M:%S.%fZ")
                            parsed_timestamps.append(dt)
                        except ValueError:
                            logger.warning(f"Invalid historical timestamp: {ts}")
                            continue
            max_timestamp = max(parsed_timestamps) if parsed_timestamps else datetime.utcnow()
            like_timestamp = max_timestamp + timedelta(days=2)
            like_timestamp_str = like_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3] + "Z"
        except Exception as e:
            logger.error(f"Error parsing timestamps: {e}")
            like_timestamp_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3] + "Z"
    else:
        like_timestamp_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3] + "Z"
    liked_asins = items
    liked_timestamps = [like_timestamp_str] * len(items)
    combined_asins = recent_asins + liked_asins
    combined_timestamps = [ts for ts in recent_timestamps if ts] + liked_timestamps
    combined_asins = combined_asins[-10:]
    combined_timestamps = combined_timestamps[-10:]
    timestamp_str = ",".join(combined_timestamps)
    item_sequence_ts = pad_timestamp_sequence(timestamp_str, sequence_length=10, padding_value=-1)
    logger.debug(f"item_sequence_ts: {item_sequence_ts}")
    try:
        last_timestamp = combined_timestamps[-1]
        try:
            dt = datetime.strptime(last_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            dt = datetime.strptime(last_timestamp[:26] + 'Z', "%Y-%m-%dT%H:%M:%S.%fZ")
        current_ts = int(dt.timestamp())
        logger.debug(f"Using current_ts: {current_ts} (from {last_timestamp})")
    except Exception as e:
        logger.error(f"Error parsing last timestamp: {e}")
        current_ts = int(time.time())
        logger.warning(f"Falling back to current time: {current_ts}")
    item_sequence_ts_bucket = calc_sequence_timestamp_bucket(item_sequence_ts, current_ts)
    logger.debug(f"item_sequence_ts_bucket: {item_sequence_ts_bucket}")
    df = pd.DataFrame.from_dict([{
        "user_id": user_id,
        "event_timestamp": datetime.utcnow(),
        "user_rating_list_10_recent_asin": ",".join(combined_asins),
        "user_rating_list_10_recent_asin_timestamp": ",".join(combined_timestamps),
        "user_rating_cnt_90d": 20,
        "user_rating_avg_prev_rating_90d": 4.3,
        "item_sequence_ts": item_sequence_ts,
        "item_sequence_ts_bucket": item_sequence_ts_bucket
    }])
    try:
        store.write_to_online_store("user_feature_view", df)
        logger.info(f"Successfully updated Feast online store for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to update Feast online store: {e}")
    logger.info(f"Recent 10 interacted ASINs for {user_id}: {combined_asins}")
    logger.info(f"Recent 10 interacted timestamps for {user_id}: {combined_timestamps}")
    logger.info(f"item_sequence_ts for {user_id}: {item_sequence_ts}")
    logger.info(f"item_sequence_ts_bucket for {user_id}: {item_sequence_ts_bucket}")
    print(f"\nRecent 10 interacted ASINs for {user_id}:")
    print(combined_asins)
    print(f"\nRecent 10 interacted timestamps for {user_id}:")
    print(combined_timestamps)
    print(f"\nitem_sequence_ts for {user_id}:")
    print(item_sequence_ts)
    print(f"\nitem_sequence_ts_bucket for {user_id}:")
    print(item_sequence_ts_bucket)
    feedback_store[user_id] = []
    product_html_cache[user_id] = {}
    html, asins, liked, buttons = render_recs(user_id)
    return f"Submitted {len(items)} liked items. List refreshed.", html, asins, liked, buttons, html

def render_single_rec(det: Dict, idx: int, is_liked: bool, user_id: str, recs: List[Dict]) -> str:
    button_value = '<i class="fas fa-heart"></i> Liked' if is_liked else '<i class="far fa-heart"></i> Like'
    button_variant = "primary" if is_liked else "secondary"
    raw_score = recs[idx].get("score", None) if idx < len(recs) else None
    score = f"{raw_score:.3f}" if isinstance(raw_score, (int, float)) else "N/A"
    return f"""
    <div class="frame rec-frame">
        <div class="frame-header">
            <img src="{det['image_url']}" class="item-img"/>
            <div class="item-info">
                <h3><i class="fas fa-star"></i> #{idx + 1} - {det['title'][:70]}{'...' if len(det['title']) > 70 else ''}</h3>
                <div class="item-meta">
                    <span class="store"><i class="fas fa-store"></i> <b>Store:</b> {det['store']}</span>
                    <span class="price"><i class="fas fa-dollar-sign"></i> <b>Price:</b> {det['price']}</span>
                </div>
                <p class="asin"><i class="fas fa-barcode"></i> <b>ASIN:</b> {det['asin']}</p>
                <p class="rating"><i class="fas fa-star-half-alt"></i> <b>Rating:</b> {det['rating']} ({det['reviews']} reviews)</p>
                <p class="categories"><i class="fas fa-tags"></i> <b>Categories:</b> {', '.join(det['categories'][:3])}</p>
                <p class="score"><i class="fas fa-chart-line"></i> <b>Recommendation Score:</b> {score}</p>
                <button class="like-button {button_variant}" data-index="{idx}" onclick="document.getElementById('hidden-like-btn-{idx}').click()">{button_value}</button>
            </div>
        </div>
    </div>
    """

def render_single_purchased(det: Dict, timestamp: str) -> str:
    formatted_time = timestamp
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            dt = datetime.strptime(timestamp[:26] + 'Z', "%Y-%m-%dT%H:%M:%S.%fZ")
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            logger.warning(f"Invalid timestamp format for purchased item: {timestamp}")
    return f"""
    <div class="frame purchased-frame">
        <div class="frame-header">
            <img src="{det['image_url']}" class="item-img"/>
            <div class="item-info">
                <h3><i class="fas fa-shopping-cart"></i> {det['title'][:70]}{'...' if len(det['title']) > 70 else ''}</h3>
                <div class="item-meta">
                    <span class="store"><i class="fas fa-store"></i> <b>Store:</b> {det['store']}</span>
                    <span class="price"><i class="fas fa-dollar-sign"></i> <b>Price:</b> {det['price']}</span>
                </div>
                <p class="asin"><i class="fas fa-barcode"></i> <b>ASIN:</b> {det['asin']}</p>
                <p class="rating"><i class="fas fa-star-half-alt"></i> <b>Rating:</b> {det['rating']} ({det['reviews']} reviews)</p>
                <p class="categories"><i class="fas fa-tags"></i> <b>Categories:</b> {', '.join(det['categories'][:3])}</p>
                <p class="timestamp"><i class="fas fa-clock"></i> <b>Purchased:</b> {formatted_time}</p>
            </div>
        </div>
    </div>
    """

def render_recs(user_id: str):
    recs = get_recommendations(user_id)
    purchased = get_purchased_products(user_id)
    product_html_cache[user_id] = {'recs': [], 'purchased': []}
    asins, liked_states, buttons = [], [], []
    rec_html = []
    if not recs:
        rec_html.append("<p>No recommendations available.</p>")
    else:
        for i, rec in enumerate(recs):
            asin = rec.get("item_id", "")
            det = get_product_details(asin)
            is_liked = asin in feedback_store.get(user_id, [])
            html = render_single_rec(det, i, is_liked, user_id, recs)
            rec_html.append(html)
            asins.append(asin)
            liked_states.append(is_liked)
            buttons.append({"value": '<i class="fas fa-heart"></i> Liked' if is_liked else '<i class="far fa-heart"></i> Like', "variant": "primary" if is_liked else "secondary" , "index": i})
    purchased_html = []
    if not purchased:
        purchased_html.append("<p>No purchased products.</p>")
    else:
        for item in purchased:
            det = get_product_details(item['asin'])
            html = render_single_purchased(det, item['timestamp'])
            purchased_html.append(html)
    product_html_cache[user_id]['recs'] = rec_html
    product_html_cache[user_id]['purchased'] = purchased_html
    html = f"""
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    </head>
    <div class='container'>
        <div class='column rec-column'>
            <h2><i class="fas fa-lightbulb"></i> Recommended Products</h2>
            <p class='section-desc'>Products recommended by the system based on user history.</p>
            <div class='frames'>{"".join(rec_html)}</div>
        </div>
        <div class='column purchased-column'>
            <h2><i class="fas fa-history"></i> Purchase History (Newest → Oldest)</h2>
            <p class='section-desc'>List of purchased products, sorted by purchase date.</p>
            <div class='frames'>{"".join(purchased_html)}</div>
        </div>
    </div>
    """
    return html, asins, liked_states, buttons

# —— CSS ——
css = """
body { 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
    background: #f9fafb;
    color: #1f2937;
}

.gradio-container { 
    max-width: 1400px; 
    margin: 0 auto; 
    padding: 40px 20px;
}

h1 { 
    text-align: center; 
    color: #111827; 
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 48px; 
}

h2 {
    font-size: 28px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-desc {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 20px;
}

.container {
    display: flex;
    gap: 40px;
    flex-wrap: wrap;
}

.column {
    flex: 1;
    min-width: 450px;
}

.rec-column {
    background: #ffffff;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.purchased-column {
    background: #f8fafc;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.frames {
    display: grid;
    grid-template-columns: 1fr;
    gap: 28px;
    padding: 0;
}

.frame {
    background: #ffffff;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 6px 16px rgba(0,0,0,0.05);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.frame:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
}

.rec-frame {
    border-left: 4px solid #3b82f6;
}

.purchased-frame {
    border-left: 4px solid #10b981;
}

.frame-header {
    display: flex;
    padding: 28px;
    gap: 28px;
    align-items: flex-start;
}

.item-img {
    width: 180px;
    height: 180px;
    object-fit: contain;
    border-radius: 12px;
    flex-shrink: 0;
    background: #f3f4f6;
}

.item-info {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.item-info h3 {
    margin: 0;
    font-size: 22px;
    font-weight: 600;
    color: #111827;
    line-height: 1.4;
    display: flex;
    align-items: center;
    gap: 8px;
}

.item-meta {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

.store, .price, .asin, .timestamp, .score {
    font-size: 15px;
    color: #4b5563;
    display: flex;
    align-items: center;
    gap: 6px;
}

.asin b, .timestamp b, .score b {
    color: #1f2937;
}

.rating, .categories {
    margin: 0;
    font-size: 15px;
    color: #4b5563;
    line-height: 1.5;
    display: flex;
    align-items: center;
    gap: 6px;
}

.rating b, .categories b {
    color: #1f2937;
}

.like-button {
    font-size: 15px;
    font-weight: 500;
    padding: 12px 32px;
    border: none;
    border-radius: 9999px;
    cursor: pointer;
    background: #e5e7eb;
    color: #374151;
    width: fit-content;
    margin-top: 12px;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}

.like-button:hover {
    background: #d1d5db;
}

.like-button.primary {
    background: #ef4444;
    color: white;
}

.like-button.primary:hover {
    background: #dc2626;
}

.message-area {
    margin-top: 40px;
    padding: 20px 28px;
    background: #f3f4f6;
    border-radius: 12px;
    border-left: 4px solid #10b981;
    color: #1f2937;
    font-size: 16px;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}

.message-area::before {
    content: '\f058';
    font-family: 'Font Awesome 5 Free';
    font-weight: 900;
    color: #10b981;
}

.control-panel {
    display: flex;
    gap: 20px;
    margin-bottom: 40px;
    align-items: center;
    flex-wrap: wrap;
}

.dropdown {
    flex: 1;
    min-width: 250px;
}

button {
    font-weight: 500;
    padding: 12px 24px;
    border-radius: 8px;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}
"""

# —— Gradio UI ——
with gr.Blocks(css=css) as demo:
    # gr.Markdown("## <i class='fas fa-gamepad'></i> Toys & Games Recommendation System")
    gr.Markdown("""
        <div style="text-align: center; margin-bottom: 32px;">
            <h1 style="
                font-size: 40px;
                font-weight: 800;
                color: #111827;
                margin-bottom: 12px;
            ">
                🧸 <span style="color: #3b82f6;">Toys & Games</span> Recommendation Dashboard
            </h1>
            <p style="
                font-size: 18px;
                color: #6b7280;
                max-width: 720px;
                margin: 0 auto;
            ">
                Stakeholder view to review AI-generated product suggestions and submit live feedback to improve personalization quality.
            </p>
        </div>
    """)


    with gr.Row(elem_classes=["control-panel"]):
        user_dd = gr.Dropdown(user_list, label="Select User", value=user_list[0], elem_classes=["dropdown"])
        btn_rec = gr.Button("Get Recommendations", variant="primary")
        btn_sub = gr.Button("Submit Likes", variant="secondary")

    out_html = gr.HTML()
    out_msg = gr.Markdown(elem_classes=["message-area"])

    like_btns = [gr.Button(visible=False, elem_id=f"hidden-like-btn-{i}") for i in range(10)]

    st_user = gr.State("")
    st_asins = gr.State([])
    st_liked = gr.State([])
    st_btns = gr.State([])

    def show_recs(user):
        feedback_store[user] = []
        product_html_cache[user] = {}
        html, asins, liked, buttons = render_recs(user)
        return html, asins, liked, buttons, user, ""

    def on_like(idx, user, asins, liked, btns):
        if idx >= len(asins):
            return "", liked, btns, "", ""
        asin = asins[idx]
        msg, new_liked, new_buttons, rec_html, purchased_html = toggle_like(user, asin, liked, idx, btns)
        html = f"""
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        </head>
        <div class='container'>
            <div class='column rec-column'>
                <h2><i class="fas fa-lightbulb"></i> Recommended Products</h2>
                <p class='section-desc'>Products recommended by the system based on user history.</p>
                <div class='frames'>{rec_html}</div>
            </div>
            <div class='column purchased-column'>
                <h2><i class="fas fa-history"></i> Purchase History (Newest → Oldest)</h2>
                <p class='section-desc'>List of purchased products, sorted by purchase date.</p>
                <div class='frames'>{purchased_html}</div>
            </div>
        </div>
        """
        return html, msg, new_liked, new_buttons

    btn_rec.click(
        fn=show_recs,
        inputs=user_dd,
        outputs=[out_html, st_asins, st_liked, st_btns, st_user, out_msg]
    )

    for i, btn in enumerate(like_btns):
        btn.click(
            fn=lambda user, asins, liked, btns, idx=i: on_like(idx, user, asins, liked, btns),
            inputs=[st_user, st_asins, st_liked, st_btns],
            outputs=[out_html, out_msg, st_liked, st_btns]
        )

    btn_sub.click(
        fn=submit_likes,
        inputs=st_user,
        outputs=[out_msg, out_html, st_asins, st_liked, st_btns, out_html]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
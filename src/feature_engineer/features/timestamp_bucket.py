import time


def bucketize_seconds_diff(seconds: int) -> int:
    """Convert a time difference in seconds to a discrete bucket index.

    Args:
        seconds (int): Time difference in seconds.

    Returns:
        int: Bucket index based on the time difference, ranging from 0 to 9.
             Buckets represent increasing time ranges:
             - 0: < 10 minutes
             - 1: 10 minutes to < 1 hour
             - 2: 1 hour to < 1 day
             - 3: 1 day to < 1 week
             - 4: 1 week to < 1 month
             - 5: 1 month to < 1 year
             - 6: 1 year to < 3 years
             - 7: 3 years to < 5 years
             - 8: 5 years to < 10 years
             - 9: >= 10 years
    """
    if seconds < 60 * 10:  # 10 minutes
        return 0
    if seconds < 60 * 60:  # 1 hour
        return 1
    if seconds < 60 * 60 * 24:  # 1 day
        return 2
    if seconds < 60 * 60 * 24 * 7:  # 1 week
        return 3
    if seconds < 60 * 60 * 24 * 30:  # 1 month
        return 4
    if seconds < 60 * 60 * 24 * 365:  # 1 year
        return 5
    if seconds < 60 * 60 * 24 * 365 * 3:  # 3 years
        return 6
    if seconds < 60 * 60 * 24 * 365 * 5:  # 5 years
        return 7
    if seconds < 60 * 60 * 24 * 365 * 10:  # 10 years
        return 8
    return 9


def from_ts_to_bucket(ts: int, current_ts: int = None) -> int:
    """Calculate the time difference bucket for a given timestamp relative to the current or provided timestamp.

    Args:
        ts (int): Unix timestamp to bucketize.
        current_ts (int, optional): Reference Unix timestamp. Defaults to current time if None.

    Returns:
        int: Bucket index calculated by bucketize_seconds_diff based on the time difference.
    """
    if current_ts is None:
        current_ts = int(time.time())
    return bucketize_seconds_diff(current_ts - ts)


def calc_sequence_timestamp_bucket(row: dict) -> list:
    """Convert a sequence of timestamps in a row to their corresponding time difference buckets.

    Args:
        row (dict): Dictionary containing 'timestamp_unix' (reference timestamp) and
                    'item_sequence_ts' (list of timestamps or -1 for padding).

    Returns:
        list: List of bucket indices for each timestamp in item_sequence_ts, preserving -1 for padding.
    """
    ts = row["timestamp_unix"]
    output = []
    for x in row["item_sequence_ts"]:
        x_i = int(x)
        if x_i == -1:
            # Keep padding (blank) element
            output.append(x_i)
        else:
            bucket = from_ts_to_bucket(x_i, ts)
            output.append(bucket)
    return output

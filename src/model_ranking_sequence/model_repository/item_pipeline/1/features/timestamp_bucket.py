import time


def bucketize_seconds_diff(seconds: int):
    if seconds < 60 * 10:
        return 0
    if seconds < 60 * 60:
        return 1
    if seconds < 60 * 60 * 24:
        return 2
    if seconds < 60 * 60 * 24 * 7:
        return 3
    if seconds < 60 * 60 * 24 * 30:
        return 4
    if seconds < 60 * 60 * 24 * 365:
        return 5
    if seconds < 60 * 60 * 24 * 365 * 3:
        return 6
    if seconds < 60 * 60 * 24 * 365 * 5:
        return 7
    if seconds < 60 * 60 * 24 * 365 * 10:
        return 8
    return 9


def from_ts_to_bucket(ts, current_ts: int = None):
    if current_ts is None:
        current_ts = int(time.time())
    return bucketize_seconds_diff(current_ts - ts)


def calc_sequence_timestamp_bucket(row):
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

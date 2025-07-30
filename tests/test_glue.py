import sys
import types
import pytest
from datetime import datetime

fake = types.ModuleType("fake")

def fake_getResolvedOptions(argv, keys):
    return {
        'JOB_NAME': 'test',
        'S3_BUCKET': 'test-bucket',
        'DB_CONNECTION': 'postgres://user:pass@host/db',
        'TABLE_NAME': 'test_table',
        'AWS_REGION': 'ap-southeast-1',
        'DB_USERNAME': 'test_user',
        'DB_PASSWORD': 'test_pw'
    }

class Dummy:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self
    def __setattr__(self, name, value):
        pass
    def __getitem__(self, key):
        return self
    def __iter__(self):
        return iter([])
    def toDF(self):
        return self
    def filter(self, *args, **kwargs):
        return self
    def show(self, *args, **kwargs):
        return None
    def count(self):
        return 0
    def dropDuplicates(self, *args, **kwargs):
        return self
    def withColumn(self, *args, **kwargs):
        return self
    def select(self, *args, **kwargs):
        return self
    def groupBy(self, *args, **kwargs):
        return self
    def orderBy(self, *args, **kwargs):
        return self
    def agg(self, *args, **kwargs):
        return self
    def collect(self):
        return []
    def alias(self, *args, **kwargs):
        return self
    def flatten(self, *args, **kwargs):
        return self
    def distinct(self):
        return self
    def join(self, *args, **kwargs):
        return self
    def first(self, *args, **kwargs):
        return self
    def avg(self, *args, **kwargs):
        return self

DummyInstance = Dummy()

def dummy_func(*args, **kwargs):
    return DummyInstance

fake.GlueContext = Dummy
fake.Job = Dummy
fake.DynamicFrame = Dummy
fake.EvaluateDataQuality = Dummy
fake.getResolvedOptions = lambda argv, keys: {
    'JOB_NAME': 'test',
    'S3_BUCKET': 'test-bucket',
    'DB_CONNECTION': 'postgres://user:pass@host/db',
    'TABLE_NAME': 'test_table',
    'AWS_REGION': 'ap-southeast-1',
    'DB_USERNAME': 'test_user',
    'DB_PASSWORD': 'test_pw'
}

for mod in [
    "awsglue", "awsglue.context", "awsglue.job", "awsglue.utils",
    "awsglue.dynamicframe", "awsgluedq", "awsgluedq.transforms", "boto3"
]:
    sys.modules[mod] = fake

sys.modules['awsglue.utils'].getResolvedOptions = fake.getResolvedOptions

from glue import (
    convert_categories,
    bucketize_seconds_diff,
    pad_timestamp_sequence,
    calc_sequence_timestamp_bucket,
)

# Test convert_categories
@pytest.mark.parametrize(
    "input_val, expected",
    [
        (None, []),
        ("", []),
        ('{"A","B"}', ["A", "B"]),
        ('["A", "B"]', ["A", "B"]),
        ({"A", "B"}, ["A", "B"]),
        (["A", "B"], ["A", "B"]),
        ('A', ["A"]),
        (123, ["123"]),
    ]
)
def test_convert_categories(input_val, expected):
    result = convert_categories(input_val)
    assert set(result) == set(expected)

# Test bucketize_seconds_diff
@pytest.mark.parametrize(
    "seconds, expected",
    [
        (60*5, 0),                 # 5 minute
        (60*30, 1),                # 30 minute
        (60*60*3, 2),              # 3 hour
        (60*60*24*3, 3),           # 3 days
        (60*60*24*15, 4),          # 15 days
        (60*60*24*180, 5),         # 180 days
        (60*60*24*366, 6),         # 1 years 1 days
        (60*60*24*366*4, 7),       # ~4 years
        (60*60*24*366*8, 8),       # ~8 years
        (60*60*24*366*12, 9),      # ~12 years
    ]
)
def test_bucketize_seconds_diff(seconds, expected):
    assert bucketize_seconds_diff(seconds) == expected

# Test pad_timestamp_sequence
def test_pad_timestamp_sequence_correct_format():
    inp = "2024-06-19T23:51:33.123Z,2023-06-20T10:00:00.000Z"
    out = pad_timestamp_sequence(inp, sequence_length=3)
    assert len(out) == 3
    # 2 timestamp, pad 1 -1 at đầu
    assert out[0] == -1
    assert isinstance(out[1], int)
    assert isinstance(out[2], int)

def test_pad_timestamp_sequence_empty():
    out = pad_timestamp_sequence("", sequence_length=5)
    assert out == [-1]*5

def test_pad_timestamp_sequence_incorrect_format():
    inp = "not-a-date,also-wrong"
    out = pad_timestamp_sequence(inp, sequence_length=2)
    assert out == [-1, -1]

# Test calc_sequence_timestamp_bucket
def test_calc_sequence_timestamp_bucket_normal():
    now = int(datetime(2024, 6, 21).timestamp())
    timestamps = [now - 100, now - 80000, now - 800000]
    current_ts = now * 1000
    buckets = calc_sequence_timestamp_bucket(timestamps, current_ts)
    assert len(buckets) == 3
    assert all(isinstance(b, int) for b in buckets)

def test_calc_sequence_timestamp_bucket_none():
    assert calc_sequence_timestamp_bucket([], 10000000) == [-1]*10
    assert calc_sequence_timestamp_bucket(None, 10000000) == [-1]*10
    assert calc_sequence_timestamp_bucket([1, -1, 2], None) == [-1]*10

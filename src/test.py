# %%
import gzip
import struct
import numpy as np
from collections import defaultdict

IMAGE_FEATURES_DATA_PATH = "data/Behance_Image_Features.b"
ITEMS_TO_OWNERS_DATA_PATH = "data/Behance_Item_to_Owners.gz"
APPRECIATE_DATA_PATH = "data/Behance_appreciate_1M.gz"
IMAGE_FEATURE_LIMIT = 70000  # set to None to load all features 

def process_gzipped_text_file(path):
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                segments = tuple(line.strip().split())
                yield segments
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    itemId = f.read(8)
    if not itemId or len(itemId) < 8: break
    feature = struct.unpack('f'*4096, f.read(4*4096))
    yield itemId, feature

g1 = readImageFeatures(path=IMAGE_FEATURES_DATA_PATH)
g2 = process_gzipped_text_file(path=ITEMS_TO_OWNERS_DATA_PATH)
g3 = process_gzipped_text_file(path=APPRECIATE_DATA_PATH)

# Item ownership lookups 
item_to_owner = {}
owner_to_items = defaultdict(set)
for row in g2:
    item, owner = row[0], row[1]
    item_to_owner[item] = owner
    owner_to_items[owner].add(item)

# Interaction histories 
user_to_items = defaultdict(list)  # user -> list of (item, timestamp)
item_to_users = defaultdict(list)  # item -> list of (user, timestamp)
for user_id, item_id, ts in g3:
    ts_int = int(ts) if ts.isdigit() else ts
    user_to_items[user_id].append((item_id, ts_int))
    item_to_users[item_id].append((user_id, ts_int))

print(f"items with owners: {len(item_to_owner):,}")
print(f"owners with items: {len(owner_to_items):,}")
print(f"users with interactions: {len(user_to_items):,}")
print(f"items with interactions: {len(item_to_users):,}")

def _decode_item_id(raw_id):
    if isinstance(raw_id, (bytes, bytearray)):
        return raw_id.decode("utf-8")
    return str(raw_id)


def load_item_feature_dict(path, limit=IMAGE_FEATURE_LIMIT):
    item_features = {}
    for idx, (raw_id, feature_tuple) in enumerate(readImageFeatures(path)):
        item_id = _decode_item_id(raw_id)
        item_features[item_id] = np.asarray(feature_tuple, dtype=np.float32)
        if limit and idx + 1 >= limit:
            break
    return item_features


def recommend_items_for_user(user_id, item_features, user_history, top_k=10, recent_n=20):
    history = user_history.get(user_id)
    if not history:
        return []

    # Use the most recent interactions to build a user profile vector.
    recent_items = [iid for iid, ts in sorted(history, key=lambda x: x[1], reverse=True)][:recent_n]
    vectors = [item_features[iid] for iid in recent_items if iid in item_features]
    if not vectors:
        return []

    profile = np.mean(np.stack(vectors, axis=0), axis=0)
    profile_norm = np.linalg.norm(profile) + 1e-8

    seen = set(iid for iid, _ in history)
    scores = []
    for item_id, vec in item_features.items():
        if item_id in seen:
            continue
        denom = (np.linalg.norm(vec) * profile_norm) + 1e-8
        score = float(np.dot(profile, vec) / denom)
        scores.append((score, item_id))

    scores.sort(reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    # Load a manageable subset of image features; adjust IMAGE_FEATURE_LIMIT as needed.
    item_features = load_item_feature_dict(IMAGE_FEATURES_DATA_PATH, limit=IMAGE_FEATURE_LIMIT)
    print(f"loaded item features: {len(item_features):,}")

    # Pick an arbitrary user to demo recommendations.
    sample_user = next(iter(user_to_items)) if user_to_items else None
    if sample_user:
        recs = recommend_items_for_user(sample_user, item_features, user_to_items, top_k=5, recent_n=10)
        print(f"sample user: {sample_user}")
        for rank, (score, item_id) in enumerate(recs, start=1):
            print(f"{rank:02d}. {item_id} (score={score:.4f})")
    else:
        print("No user interactions available to generate recommendations.")
from modelscope.hub.snapshot_download import snapshot_download
import os

MODEL_ID = "openai-mirror/clip-vit-base-patch32"   # ModelScope 上的常见命名
CACHE_DIR = "/home/qmn/Yuan-chat/MPSE_FULL_LOOP_PROJECT/models/clip-vit-base-patch32"  # 你想放哪儿就改哪儿

os.makedirs(CACHE_DIR, exist_ok=True)
local_dir = snapshot_download(model_id=MODEL_ID, cache_dir=CACHE_DIR)
print("Downloaded to:", local_dir)

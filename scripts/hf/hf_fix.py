import os, sys, time, logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("/root/hf_fix.log"), logging.StreamHandler()],
)
log = logging.getLogger("hf_fix")

REPO = "ZhenYaoWang/sol-h3-fp8-5090"
HF8 = "/root/hyperflow/transformer_hf8"
REF = "/root/hyperflow/transformer_ref_hf8"
BACKUP = "/root/hyperflow/table-backups"

from huggingface_hub import HfApi
api = HfApi()

def step(msg):
    log.info("==== %s ====", msg)

# 0. 再次保险：把任何 .bak-audio 移出待传目录
step("0. 清理 .bak-audio（保险）")
for d in (HF8, REF):
    for f in Path(d).glob("*.bak-audio"):
        dest = Path(BACKUP) / f.name
        if not dest.exists():
            f.rename(dest)
        else:
            f.unlink()
        log.info("moved %s", f.name)

# 1. 上传主 DiT 到 transformer_hf8
step("1. 上传 transformer_hf8 -> repo/transformer_hf8")
t0 = time.time()
api.upload_folder(
    folder_path=HF8,
    repo_id=REPO,
    repo_type="model",
    path_in_repo="transformer_hf8",
)
log.info("主 DiT 上传完成，耗时 %.1f 分钟", (time.time() - t0) / 60)

# 2. 上传参考图 DiT 到 transformer_ref_hf8
step("2. 上传 transformer_ref_hf8 -> repo/transformer_ref_hf8")
t0 = time.time()
api.upload_folder(
    folder_path=REF,
    repo_id=REPO,
    repo_type="model",
    path_in_repo="transformer_ref_hf8",
)
log.info("参考图 DiT 上传完成，耗时 %.1f 分钟", (time.time() - t0) / 60)

# 3. 删除根目录散装文件（覆盖事故残留）
step("3. 删除根目录散装文件")
files = api.list_repo_files(REPO, repo_type="model")
root_files = [f for f in files if "/" not in f]
log.info("根目录散装文件共 %d 个：%s", len(root_files), root_files)
for f in root_files:
    try:
        api.delete_file(path_in_repo=f, repo_id=REPO, repo_type="model")
        log.info("deleted %s", f)
    except Exception as e:
        log.warning("删除 %s 失败: %s", f, e)

# 4. 验证结构
step("4. 验证最终结构")
files = api.list_repo_files(REPO, repo_type="model")
subdirs = sorted({f.split("/", 1)[0] for f in files if "/" in f})
root_left = [f for f in files if "/" not in f]
log.info("子目录: %s", subdirs)
log.info("根目录残留: %s", root_left)
log.info("文件总数: %d", len(files))
print("SUBDIRS", subdirs)
print("ROOT_LEFT", root_left)
log.info("==== 全部完成 ====")

import subprocess, hashlib, os, json
# 1) 缓存文件状态
print("=== cache file /root/.hf_shas.json ===")
print("exists:", os.path.exists("/root/.hf_shas.json"), "size:", os.path.getsize("/root/.hf_shas.json") if os.path.exists("/root/.hf_shas.json") else "n/a")
if os.path.exists("/root/.hf_shas.json"):
    with open("/root/.hf_shas.json") as f:
        c = json.load(f)
    print("cache keys:", list(c.keys()))
    for k in c:
        print("  ", k, "分片数:", len(c[k]))
# 2) curl tree，取第一个 safetensors 的完整原始条目
out = subprocess.run(["curl", "-s", "--max-time", "60",
    "https://huggingface.co/api/models/ZhenYaoWang/sol-h3-fp8-5090/tree/main?recursive=true&expand=false"],
    capture_output=True, text=True)
tree = json.loads(out.stdout)
print("=== tree 解析: 共", len(tree), "条目 ===")
target = None
for it in tree:
    p = it.get("path", "")
    if p.endswith(".safetensors") and "index" not in p and p.count("/") == 1:
        target = it
        break
if target:
    print("选中的文件:", target.get("path"))
    print("原始 oid:", target.get("oid"))
    print("原始 lfs:", json.dumps(target.get("lfs")))
    lfs_sha = (target.get("lfs") or {}).get("sha256")
    print("lfs.sha256 =", lfs_sha)
    # 3) 本地该分片真实 sha256
    local_path = os.path.join("/root/hyperflow", target.get("path"))
    if os.path.exists(local_path):
        h = hashlib.sha256()
        with open(local_path, "rb") as f:
            for c in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(c)
        local_sha = h.hexdigest()
        print("本地 sha256 =", local_sha)
        print(">>> 线上 lfs.sha256 == 本地 sha256 ?", local_sha == lfs_sha)
    else:
        print("!! 本地源文件不存在:", local_path)
else:
    print("!! tree 里没找到 safetensors 分片")

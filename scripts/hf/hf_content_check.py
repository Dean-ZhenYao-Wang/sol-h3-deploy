import hashlib, os, json, subprocess, time
REPO = "ZhenYaoWang/sol-h3-fp8-5090"
SOURCES = {"transformer_hf8": "/root/hyperflow/transformer_hf8",
           "transformer_ref_hf8": "/root/hyperflow/transformer_ref_hf8"}
CACHE = "/root/.hf_shas.json"
LOG = "/root/hf_content_check.log"

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

t0 = time.time()
with open(LOG, "w", buffering=1) as log:
    log.write("==== 1) 本地源分片 sha256 ====\n")
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            local_shas = json.load(f)
        log.write("从缓存读取本地 sha256 (跳过 45G 重算)\n")
    else:
        local_shas = {}
        for sub, src in SOURCES.items():
            local_shas[sub] = {}
            for fn in sorted(os.listdir(src)):
                if fn.endswith(".safetensors") and "index" not in fn:
                    s = sha256_of(os.path.join(src, fn))
                    local_shas[sub][fn] = s
                    log.write(f"{sub}/{fn}: {s[:16]}...\n")
        with open(CACHE, "w") as f:
            json.dump(local_shas, f)
        log.write("本地 sha256 已缓存到 %s\n" % CACHE)

    inter = set(local_shas["transformer_hf8"].values()) & set(local_shas["transformer_ref_hf8"].values())
    log.write(f"==== 2) 两份源权重是否不同: 分片 sha256 交集={len(inter)} (应为0) ====\n")

    log.write("==== 3) 线上 LFS sha256 (curl tree API) ====\n")
    url = f"https://huggingface.co/api/models/{REPO}/tree/main?recursive=true&expand=false"
    out = subprocess.run(["curl", "-s", "--max-time", "90", url], capture_output=True, text=True)
    try:
        tree = json.loads(out.stdout)
    except Exception as e:
        log.write("!! tree API 解析失败: %s\n" % e)
        tree = []
    online = {"transformer_hf8": {}, "transformer_ref_hf8": {}}
    for item in tree:
        p = item.get("path", "")
        if p.count("/") == 1:
            sub, fn = p.split("/", 1)
            if sub in online and fn.endswith(".safetensors") and "index" not in fn:
                lfs = item.get("lfs")
                # LFS 的文件内容 sha256 在 lfs.oid（不是 lfs.sha256）
                sha = (lfs.get("sha256") or lfs.get("oid")) if lfs else None
                online[sub][fn] = sha
                log.write(f"online {p}: {sha[:16] if sha else 'None'}...\n")

    log.write("==== 4) 比对结论 ====\n")
    ok = True
    for sub in SOURCES:
        got, exp = online[sub], local_shas[sub]
        names = set(got) & set(exp)
        match = sum(1 for n in names if got[n] and got[n] == exp[n])
        log.write(f"{sub}: 线上分片={len(got)}, 本地分片={len(exp)}, sha256匹配={match}/{len(names)}\n")
        if match != len(names) or len(got) != len(exp):
            ok = False
    cross = sum(1 for n in (set(online["transformer_hf8"]) & set(local_shas["transformer_ref_hf8"]))
                if online["transformer_hf8"].get(n) and online["transformer_hf8"][n] == local_shas["transformer_ref_hf8"].get(n))
    log.write(f"hf8线上 == ref本地 的意外匹配: {cross} (应为0)\n")
    if cross:
        ok = False
    log.write(f"两份源不同: {len(inter)==0}\n")
    log.write("==== 总体结论: " + ("PASS 两个子目录内容不同且各自正确对应本地源" if (ok and len(inter)==0) else "FAIL 见上") + " ====\n")
    log.write(f"耗时 {time.time()-t0:.1f}s\n")
    log.write("==== DONE ====\n")

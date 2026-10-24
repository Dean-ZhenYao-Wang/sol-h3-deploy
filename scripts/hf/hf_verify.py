from huggingface_hub import HfApi
files = HfApi().list_repo_files("ZhenYaoWang/sol-h3-fp8-5090")
dirs = sorted({f.split("/")[0] for f in files if "/" in f})
root = [f for f in files if "/" not in f and not f.startswith(".")]
main = [f for f in files if f.startswith("transformer_hf8/")]
ref = [f for f in files if f.startswith("transformer_ref_hf8/")]
print("VERIFY 子目录:", dirs)
print("VERIFY 根目录残留:", root)
print("VERIFY transformer_hf8 文件数:", len(main))
print("VERIFY transformer_ref_hf8 文件数:", len(ref))
ok = dirs == ["transformer_hf8", "transformer_ref_hf8"] and not root and len(main) == 17 and len(ref) == 17
print("VERIFY_RESULT:", "PASS" if ok else "FAIL")

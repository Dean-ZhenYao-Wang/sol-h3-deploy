from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("ZhenYaoWang/sol-h3-fp8-5090", repo_type="model")
root = [f for f in files if "/" not in f and not f.startswith(".")]
print("root files to delete:", len(root))
for f in root:
    api.delete_file("ZhenYaoWang/sol-h3-fp8-5090", f)
    print("deleted", f)
print("DELETE_DONE")

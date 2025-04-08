from huggingface_hub import snapshot_download

# Download the model from Hugging Face Hub
snapshot_download(repo_id="apple/DepthPro-hf", repo_type="model", local_dir="Pretrained/depth_pro")
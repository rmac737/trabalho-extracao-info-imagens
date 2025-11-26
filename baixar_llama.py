from huggingface_hub import snapshot_download

# Baixa todo o repo para esta pasta local
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="Llama-3.1-8B-Instruct",
    local_dir_use_symlinks=False,  # melhor no Windows
)
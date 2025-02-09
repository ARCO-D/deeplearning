# 创建venv
python -m venv deepseek-env

# 随便装个torch版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets

# 下载模型
# https://hf-mirror.com/Qwen/Qwen2.5-0.5B (超小模型，仅0.5B
# https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Llama-8B （deepseek蒸馏的Llama-8B模型，运行需要32G内存
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/Qwen/Qwen2.5-3B # (float16下运行需要8G显存
cd Qwen/Qwen2.5-3B
git-lfs checkout
git-lfs ls-files --size
git-lfs pull


# load conda into environment
eval "$(conda shell.bash hook)"

# set CUDA_HOME (to PyTorch cuda version)
export CUDA_HOME=/usr/local/cuda-10.1

# make directories for apex
mkdir -p ~/lib && cd ~/lib
git clone https://github.com/NVIDIA/apex
cd apex

# install apex
conda activate cpc && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

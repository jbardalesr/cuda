# Workspace configuration
All example in this repository was made in WSL Windows 11
## Install cuda in WSL
[Official NVIDIA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#ch03a-setting-up-cuda)

After installing WSL with the command `wsl --install` run the following commands

1. Install CUDA:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
2. If `nvcc` compiler doesn't work, run this command obtained from [Ask ubuntu](https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed)
```
nano /home/$USER/.bashrc
# inside of the file .bashrc
export PATH="/usr/local/cuda-11.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"

# after saving and exiting run the following command
source .bashrc

# check
nvcc --version
```

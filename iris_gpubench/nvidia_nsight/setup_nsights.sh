#! /bin/bash

# Set up system
cat /proc/sys/kernel/perf_event_paranoid

sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'

sudo sh -c 'echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf'

# Install Nsights Package Manager for Ubuntu
sudo apt update
sudo apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt update
sudo apt install nsight-systems-cli

# Check CLI is set up correctly

nsys status -e

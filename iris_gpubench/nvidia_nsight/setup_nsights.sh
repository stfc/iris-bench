#! /bin/bash

# Set up system (changes kernel paranoid level untill reboot)
sudo sysctl kernel.perf_event_paranoid=0


# Install Nsights Package Manager for Ubuntu
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
sudo apt install nsight-systems

# Check CLI is set up correctly

nsys status -e

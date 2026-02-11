#!/bin/bash

# =============================================================================
# setup_vm_tmux.sh
# =============================================================================
# This script sets up a virtual machine for running benchmarks and iris-gpubench
# with Tmux on Ubuntu. The Ubuntu VM should be equipped with Nvidia GPUs.
# It installs essential software packages for iris-gpubench including Git, Python, 
# Tmux, and NVIDIA drivers.
# =============================================================================

# Get Nvidia driver version from first argument to script
NVIDIA_DRIVER_VERSION=$1

if [ -z "$NVIDIA_DRIVER_VERSION" ] ;then
   echo "Please provide the Nvidia driver version. e.g. ./setup_vm_tmux.sh <nvidia-driver-version>"
   exit
fi

#### INSTALL Git, python3-pip and Venv ###

echo -e "/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/nInstall git, python3-pip, and Venv/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/n"

# Install Git, Python, and venv
sudo apt-get install -y git
sudo apt-get install -y wget python3-pip
sudo apt-get install -y python3-venv

#### INSTALL TMUX


echo -e "/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/nINSTALL Tmux/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/n"

sudo apt update
sudo apt install tmux


##### INSTALL NVIDIA DRIVERS #####

echo -e "/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/nINSTALL NVIDIA DRIVERS/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/n"

# Install NVIDIA drivers (if needed)
# Ensure the driver is up-to-date for the GPU to achieve maximum performance.
sudo apt install -y "nvidia-driver-${NVIDIA_DRIVER_VERSION}"

##### REBOOT VM #####

echo -e "/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/nREBOOT/n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/n"

# Reboot the system to finish setup
sudo reboot

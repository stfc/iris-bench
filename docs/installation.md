## Pre-Installation Instructions

1. Build a VM in OpenStack with: 
   - Image: `ubuntu-jammy-22.04-nogui` or `ubuntu-noble-24.04-nogui`
   - Flavor: Any `g-*` flavor with an NVIDIA GPU
   - Your key pair for SSH access
   - Accessible networking: Internal or Private with FIP

2. Use [this](https://www.nvidia.com/en-gb/drivers/) NVIDIA page to find the latest driver for your VMs GPU and note the number e.g. 590

3. Install Dependencies:
   ```shell
   git clone https://github.com/stfc/iris-bench.git
   cd iris-bench
   # For using Docker containers to benchmark:
   ./setup_vm_docker.sh <nvidia-driver-version>
   # For direct machine benchmarking:
   ./setup_vm_tmux.sh <nvidia-driver-version>
   ```

---

## Installation Instructions

Follow these steps to set up iris-gpubench:

### 1.**Clone the Repository**  
   Start by cloning the project repository:
```sh
git clone https://github.com/bryceshirley/iris-gpubench.git
cd iris-gpubench
```

### 2.**Set Up a Virtual Environment**  
   Next, create and activate a virtual environment:
```sh
python3 -m venv env
source env/bin/activate
```

### 3.**Install Dependencies and iris-gpubench Package**  
####   a. Finally, install the package along with necessary dependencies:
```sh
pip install wheel
pip install .
```
####   b. **(For Developers)**
```sh
pip install wheel
pip install -e .
```
   -  `-e` for editable mode, lets you install Python packages in a way that
   allows immediate reflection of any changes you make to the source code
   without needing to reinstall the package.

---

[Previous Page](overview.md) | [Index](index.md) | [Next Page](building_docker_images.md)

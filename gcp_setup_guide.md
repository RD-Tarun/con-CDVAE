# GCP Setup & Training Guide for Con-CDVAE (Dual Conditions)

This guide covers everything you need to set up and train the Con-CDVAE model with **both Formation Energy and Band Gap conditions** on Google Cloud Platform (GCP).

---

## Table of Contents

1. [Create a GCP VM Instance](#1-create-a-gcp-vm-instance)
2. [Connect to the VM](#2-connect-to-the-vm)
3. [Install NVIDIA Drivers & CUDA](#3-install-nvidia-drivers--cuda)
4. [Set Up the Conda Environment](#4-set-up-the-conda-environment)
5. [Upload Your Code & Data](#5-upload-your-code--data)
6. [Configure Environment Variables](#6-configure-environment-variables)
7. [Prepare the Dataset](#7-prepare-the-dataset)
8. [Run Training](#8-run-training)
9. [Monitor Training](#9-monitor-training)
10. [Download Results](#10-download-results)
11. [Generate Crystals](#11-generate-crystals)
12. [Cost Optimization Tips](#12-cost-optimization-tips)

---

## 1. Create a GCP VM Instance

### Option A: Via GCP Console (Web UI)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Compute Engine** → **VM Instances** → **Create Instance**
3. Configure the instance:

| Setting | Recommended Value |
|---------|-------------------|
| **Name** | `concdvae-training` |
| **Region** | `us-central1` (cheapest for GPUs) |
| **Machine Type** | `n1-standard-8` (8 vCPUs, 30 GB RAM) |
| **GPU** | 1x NVIDIA T4 (budget) or 1x V100 (faster) or 1x A100 (fastest) |
| **Boot Disk** | Ubuntu 22.04 LTS, 100 GB SSD |
| **Firewall** | Allow HTTP/HTTPS traffic |

4. Click **Create**

### Option B: Via `gcloud` CLI

```bash
# Install gcloud CLI first: https://cloud.google.com/sdk/docs/install

# Create VM with T4 GPU (budget option, ~$0.35/hr)
gcloud compute instances create concdvae-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"

# OR: Create VM with V100 GPU (faster, ~$2.48/hr)
gcloud compute instances create concdvae-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"

# OR: Create VM with A100 GPU (fastest, ~$3.67/hr)
gcloud compute instances create concdvae-training \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE
```

### GPU Comparison

| GPU | VRAM | Training Speed | Cost/hr (approx) | Recommended For |
|-----|------|----------------|-------------------|-----------------|
| T4 | 16 GB | ~1x (baseline) | $0.35 | Budget training, small datasets |
| V100 | 16 GB | ~2-3x faster | $2.48 | Medium datasets, faster iteration |
| A100 | 40 GB | ~4-5x faster | $3.67 | Large datasets, full mp_20 training |

---

## 2. Connect to the VM

```bash
# SSH into the VM
gcloud compute ssh concdvae-training --zone=us-central1-a

# Or use SSH with port forwarding (for monitoring)
gcloud compute ssh concdvae-training --zone=us-central1-a -- -L 6006:localhost:6006
```

---

## 3. Install NVIDIA Drivers & CUDA

If you used `--metadata="install-nvidia-driver=True"`, drivers should auto-install. Otherwise:

```bash
# Check if GPU is detected
nvidia-smi

# If not installed, install CUDA toolkit
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit

# Reboot and verify
sudo reboot
# After reboot, SSH back in and run:
nvidia-smi
```

You should see your GPU listed (T4/V100/A100).

---

## 4. Set Up the Conda Environment

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# Create the conda environment
conda create --name concdvae310 python=3.10 -y
conda activate concdvae310

# Install PyTorch with CUDA support
# Check your CUDA version with: nvidia-smi (look at top right for CUDA Version)
# For CUDA 11.8:
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 \
#     --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric and extensions
pip install torch_geometric==2.5.3
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

# Install PyTorch Lightning
pip install pytorch-lightning==2.4.0 torchmetrics==1.6.3

# Install remaining requirements
cd ~/con-CDVAE   # your project directory
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 5. Upload Your Code & Data

### Option A: Using `gcloud scp` (from local machine)

```bash
# From your LOCAL Windows machine (PowerShell/CMD):
# Upload the entire project
gcloud compute scp --recurse "d:\Works\FinalYrProject\New folder (2)\con-CDVAE" \
    concdvae-training:~/con-CDVAE --zone=us-central1-a

# Or upload just the specific folders
gcloud compute scp --recurse "d:\Works\FinalYrProject\New folder (2)\con-CDVAE\concdvae" \
    concdvae-training:~/con-CDVAE/concdvae --zone=us-central1-a
gcloud compute scp --recurse "d:\Works\FinalYrProject\New folder (2)\con-CDVAE\conf" \
    concdvae-training:~/con-CDVAE/conf --zone=us-central1-a
gcloud compute scp --recurse "d:\Works\FinalYrProject\New folder (2)\con-CDVAE\scripts" \
    concdvae-training:~/con-CDVAE/scripts --zone=us-central1-a
gcloud compute scp --recurse "d:\Works\FinalYrProject\New folder (2)\con-CDVAE\data" \
    concdvae-training:~/con-CDVAE/data --zone=us-central1-a
```

### Option B: Using Git

```bash
# On the GCP VM:
cd ~
git clone <your-repo-url> con-CDVAE
cd con-CDVAE
```

### Option C: Using Google Cloud Storage (for large datasets)

```bash
# From local machine: upload to GCS bucket
gsutil -m cp -r "d:\Works\FinalYrProject\New folder (2)\con-CDVAE\data\mp_20" gs://your-bucket/concdvae/data/mp_20

# On GCP VM: download from GCS
gsutil -m cp -r gs://your-bucket/concdvae/data/mp_20 ~/con-CDVAE/data/mp_20
```

---

## 6. Configure Environment Variables

```bash
cd ~/con-CDVAE

# Create .env file
cat > .env << 'EOF'
export PROJECT_ROOT="/home/$(whoami)/con-CDVAE"
export HYDRA_JOBS="/home/$(whoami)/con-CDVAE/output/hydra"
export WABDB_DIR="/home/$(whoami)/con-CDVAE/output/wandb"
EOF

# Fix the .env to expand the actual username
sed -i "s|\$(whoami)|$(whoami)|g" .env

# Source it
source .env

# Verify
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "HYDRA_JOBS=${HYDRA_JOBS}"

# Create output directories
mkdir -p ${HYDRA_JOBS} ${WABDB_DIR}
```

---

## 7. Prepare the Dataset

### Using the sample dataset (mptest) for testing:
The `mptest` dataset is already included in the repository. The CSV files already contain both `formation_energy_per_atom` and `band_gap` columns.

### Using the full mp_20 dataset:
The **MP-20 dataset is already included** in the repository at `data/mp_20/`.
When you uploaded the code in Step 5, the dataset was uploaded automatically (unless you intentionally excluded it).

The provided training script is already configured to use these existing files:
```
data/mp_20/train.csv
data/mp_20/val.csv
data/mp_20/test.csv
```

---

## 8. Run Training

### Quick Test (with sample data, ~5 minutes):

```bash
cd ~/con-CDVAE
source .env
conda activate concdvae310

# Test run with small dataset (3 epochs)
python concdvae/run.py data=mptest_format_gap expname=test_dual model=vae_mp_format_gap
```

### Full Training (with mp_20 dataset):

```bash
cd ~/con-CDVAE
source .env
conda activate concdvae310

# OPTION 1: Use the training script (recommended)
bash scripts/train_format_gap.sh mp20_format_gap format_gap_dual 1

# OPTION 2: Run step-by-step

# Step 1: Train CDVAE (single GPU, ~24-48 hours on T4, ~8-12h on V100)
python concdvae/run.py \
    data=mp20_format_gap \
    expname=format_gap_dual \
    model=vae_mp_format_gap

# Step 2: Train Prior (after Step 1 completes)
# Replace YYYY-MM-DD with the actual date directory
python concdvae/run_prior.py \
    --model_path ${HYDRA_JOBS}/singlerun/YYYY-MM-DD/format_gap_dual \
    --prior_label prior_default
```

### Multi-GPU Training (if you have multiple GPUs):

```bash
# 4 GPU training
torchrun --nproc_per_node 4 concdvae/run.py \
    data=mp20_format_gap \
    expname=format_gap_dual \
    model=vae_mp_format_gap \
    train.pl_trainer.accelerator=gpu \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp_find_unused_parameters_true
```

### Run in Background (keep training even if SSH disconnects):

```bash
# Using nohup
nohup bash scripts/train_format_gap.sh mp20_format_gap format_gap_dual 1 \
    > training_log.txt 2>&1 &

# View live logs
tail -f training_log.txt

# OR using screen (recommended)
screen -S training
bash scripts/train_format_gap.sh mp20_format_gap format_gap_dual 1
# Press Ctrl+A then D to detach
# Reconnect later: screen -r training

# OR using tmux
tmux new -s training
bash scripts/train_format_gap.sh mp20_format_gap format_gap_dual 1
# Press Ctrl+B then D to detach
# Reconnect later: tmux attach -t training
```

---

## 9. Monitor Training

### Check GPU Usage:

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi
```

### Check Training Logs:

```bash
# Training creates CSV logs in the model directory
# Find the latest training directory:
ls -lt ${HYDRA_JOBS}/singlerun/

# View training metrics:
cat ${HYDRA_JOBS}/singlerun/YYYY-MM-DD/format_gap_dual/mp20/version_0/metrics.csv
```

### Key Metrics to Watch:
- `train_loss` — should decrease over time
- `train_predict_loss` — combines BOTH formation_energy + band_gap prediction loss
- `train_formation_energy_per_atom_loss` — formation energy prediction loss
- `train_band_gap_loss` — band gap prediction loss
- `train_coord_loss` — coordinate diffusion loss
- `val_loss` — validation loss (should decrease)

---

## 10. Download Results

### Download trained model to local machine:

```bash
# From your LOCAL machine:
# Download the full model directory
gcloud compute scp --recurse \
    concdvae-training:~/con-CDVAE/output/hydra/singlerun/YYYY-MM-DD/format_gap_dual \
    ./trained_model/ --zone=us-central1-a

# Or download just the checkpoints
gcloud compute scp \
    "concdvae-training:~/con-CDVAE/output/hydra/singlerun/YYYY-MM-DD/format_gap_dual/*.ckpt" \
    ./trained_model/ --zone=us-central1-a
```

---

## 11. Generate Crystals

After training is complete, generate crystal structures with both conditions:

```bash
# First, update the generation config with your model path:
# The model path will be printed at the end of the training script.
# Use nano to edit the file directly on the VM:
nano conf/gen/format_gap_default.yaml

# Change the `model_path:` line to:
# model_path: /home/<user>/con-CDVAE/output/hydra/singlerun/YYYY-MM-DD/format_gap_dual

# Generate crystals
python scripts/gen_crystal.py --config conf/gen/format_gap_default.yaml
```

The generation input CSV (`general_format_gap.csv`) specifies target conditions:
```csv
label,formation_energy_per_atom,band_gap
FG0,-1.0,0.0      # Low formation energy, metallic
FG1,-1.0,1.0      # Low formation energy, small band gap
FG2,-2.0,3.0      # Medium formation energy, large band gap
...
```

Generated crystal structures will be saved as `eval_gen_format_gap_FG*.pt` files in the model directory.

---

## 12. Cost Optimization Tips

### Stop the VM when not training:
```bash
# From local machine:
gcloud compute instances stop concdvae-training --zone=us-central1-a

# Restart when needed:
gcloud compute instances start concdvae-training --zone=us-central1-a
```

### Use Preemptible/Spot VMs (60-80% cheaper):
```bash
# Add --provisioning-model=SPOT when creating the VM
gcloud compute instances create concdvae-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --metadata="install-nvidia-driver=True"
```

> **⚠️ Warning**: Spot VMs can be preempted (shut down) at any time. Use checkpointing (`save_last: True` is already enabled) to resume training.

### Estimated Training Costs (mp_20 dataset, 500 epochs):

| GPU | Time (approx) | Cost (approx) |
|-----|---------------|----------------|
| T4 (standard) | 24-48 hours | $8-17 |
| T4 (spot) | 24-48 hours | $3-7 |
| V100 (standard) | 8-16 hours | $20-40 |
| V100 (spot) | 8-16 hours | $7-14 |
| A100 (standard) | 4-8 hours | $15-30 |

---

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce `batch_size` in the data config (e.g., from 256 to 128 or 64)

2. **NaN loss during training**: This is why `smooth: True` is set in the model config. If it persists, try reducing the learning rate.

3. **Missing properties in dataset**: Ensure your CSV has both `formation_energy_per_atom` and `band_gap` columns.

4. **Can't find checkpoint**: The training script automatically finds the latest checkpoint. If it fails, manually specify the path.

5. **GPU not detected**: Run `nvidia-smi` to check. If not working, reinstall drivers:
   ```bash
   sudo apt-get install -y nvidia-driver-535
   sudo reboot
   ```

6. **Permission denied on .sh files**:
   ```bash
   chmod +x scripts/train_format_gap.sh
   ```

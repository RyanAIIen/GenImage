# Swin Transformer for Image Generation Detection

This repository contains our implementation of image generation detection using
the Swin Transformer architecture. The original Swin Transformer paper and code
can be found [here](https://arxiv.org/abs/2103.14030).

## Setup

1. Create and activate conda environment:
```bash
conda create -n swin python=3.8 -y
conda activate swin
```

2. Install PyTorch with CUDA support:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
```

3. Install other dependencies:
```bash
pip install timm==0.4.12 opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```

4. Optional: Install CUDA acceleration:
```bash
cd kernels/window_process && python setup.py install && cd ../../
```

## Usage

### Training

To start training on the Midjourney dataset:

It's recommended to use `screen` to prevent training interruption if your connection drops:

```bash
# Start a new screen session
screen -S swin_training

# Activate environment
conda activate swin

# Start training
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 \
    main.py --cfg configs/ryan/swin_midjourney.yaml \
    --tag "midjourney_run1"
```

You can safely detach from the screen session with `Ctrl+A` then `D`. To reattach:
```bash
screen -r swin_training
```

If your connection drops, just reconnect and reattach to the screen session - your training will still be running.
```

The training process will:
- Save checkpoints to `output/ryan_swin_midjourney/midjourney_run1/`
- Log training metrics to TensorBoard
- Display progress every 100 iterations

The output directory structure follows:
- `output/` (root directory)
  - `ryan_swin_midjourney/` (from MODEL.NAME)
    - `midjourney_run1/` (from --tag parameter)

Use meaningful tags to organize different experiments (e.g., "midjourney_baseline", "midjourney_augmented").

You can monitor training with:
```bash
tensorboard --logdir output/ryan_swin_midjourney/midjourney_run1/tensorboard
```

### Validation

After successful training, you can validate on different datasets:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 \
    main.py --eval \
    --cfg configs/ryan/swin_midjourney.yaml \
    --resume output/ryan_swin_midjourney/midjourney_run1/checkpoint.pth \
    --data-path /nfs/turbo/umd-anglial/GenImageDetector/raw_dataset/<adm |  glide |  sd_v1_4 |  sd_v1_5 |  vqdm |  wukong>/val
```


## Model Configuration

Our configuration for binary classification is in `configs/ryan/swin_midjourney.yaml`:

Key settings:
- Model: Swin-T architecture (28M parameters)
- Training:
  - Batch size: 64
  - Epochs: 100
  - Learning rate: 5e-4
  - Mixed precision training enabled
  - Progress saved every epoch
- Data:
  - Binary classification (real/AI-generated)
  - 224x224 image size
  - Basic augmentations (color jitter)

## Development Files Structure

```
configs/
  ryan/
    swin_midjourney.yaml  # Our configuration
output/
  ryan_swin_midjourney/  # Model name directory
    midjourney_run1/            # Experiment tag directory
      checkpoint.pth            # Latest checkpoint
      tensorboard/             # Training logs
main.py                        # Training/validation script
requirements.txt               # Python dependencies
```

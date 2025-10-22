## [CNN Detection](https://peterwang512.github.io/CNNDetection/)

### Environment Setup

For training with modern CUDA GPUs (e.g., A40), use the following environment setup:
```bash
# Create conda environment
conda create -n cnn-detect python=3.8
conda activate cnn-detect

# Install PyTorch with CUDA support
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### Training on Custom Datasets

The model can be trained on custom AI-generated image datasets. For example, to
train on the Midjourney dataset:

```bash
python train.py \
    --name midjourney1 \
    --dataroot /path/to/dataset/midjourney/train
```

The final model will be saved as `checkpoints/<model_path>/model_epoch_best.pth`.
Training progress can be monitored through `checkpoints/<model_path>/log.txt`.

### Model Validation

To validate a trained model's performance, you can run:

```bash
python demo_dir.py \
    --model_path checkpoints/midjourney1/model_epoch_best.pth
    -dataroot /path/to/dataset/midjourney/val
```

This will output accuracy and precision metrics for the validation dataset to
`./checkpoints/<model_path>/validation_results_<timestamp>_<model>_<dataset>.txt`

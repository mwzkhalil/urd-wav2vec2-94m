# Urdu ASR: Wav2Vec2 CTC Fine-tuning

Fine-tuning Wav2Vec2 CTC models for Urdu Automatic Speech Recognition (ASR).

## Overview

This repository provides a pipeline for finetuning the `mahwizzzz/urd-wav2vec2-94m` model on custom Urdu speech datasets. The implementation uses HuggingFace Transformers with CTC loss, mixed precision training, and comprehensive evaluation metrics.

## Features

- **HuggingFace Trainer**: Uses official Trainer API with CTC loss
- **Data Loading**: Fault tolerant dataset handling with automatic train/val splits
- **Mixed Precision**: FP16 training for faster convergence and lower memory usage
- **Evaluation**: WER, MER, WIL, and WIP metrics
- **Checkpoint Management**: Automatic best model saving based on validation WER
- **Configurable**: All paths and hyperparameters via YAML config files

## Requirements

- Python 3.10
- CUDA-capable GPU (Tesla T4 or similar, 16GB+ VRAM recommended)
- Linux/Unix environment
- 50GB+ free disk space (for dataset and checkpoints)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/mwzkhalil/urd-wav2vec2-94m.git
cd urd-wav2vec2-94m
```

2. Create and activate virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Format

The dataset must follow this structure:

```
dataset/
├── audio/
│   ├── audio_00001.wav
│   ├── audio_00002.wav
│   └── ...
└── metadata.csv
```

### Metadata CSV Format

The `metadata.csv` file must have the following columns:

```csv
text,audio_path
تزک بابری از زہیر الدین محمد بابر,/audio_00001.wav
پیش لفظ زیر نظر کتاب تزک بابری کا مصنف,/audio_00002.wav
```

**Requirements:**
- `text`: Urdu transcription text (UTF-8 encoded)
- `audio_path`: Path to audio file (relative to dataset root or absolute)
- Audio files must be in WAV format
- Audio will be automatically resampled to 16kHz
- Maximum audio duration: 30 seconds (configurable)
- Minimum audio duration: 0.5 seconds (configurable)

## Configuration

### Data Configuration (`config/data.yaml`)

```yaml
dataset:
  metadata_path: "metadata.csv"
  audio_dir: "audio"
  max_duration: 30.0
  min_duration: 0.5
  train_ratio: 0.9
  seed: 42
```

### Training Configuration (`config/training.yaml`)

```yaml
model:
  model_name: "mahwizzzz/urd-wav2vec2-94m"

training:
  output_dir: "checkpoints"
  logging_dir: "logs"
  num_train_epochs: 10
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  warmup_steps: 500
  weight_decay: 0.01
  fp16: true
  save_strategy: "epoch"
  evaluation_strategy: "epoch"
  metric_for_best_model: "wer"
```

## Usage

### 1. Prepare Dataset

Validate your dataset structure:

```bash
python scripts/prepare_dataset.py \
    --metadata metadata.csv \
    --audio_dir audio \
    --max_duration 30.0
```

This will:
- Validate all audio files exist
- Check audio durations
- Report dataset statistics
- Create train/val splits automatically

### 2. Training

#### Option A: Using the training script

```bash
python train.py \
    --training_config config/training.yaml \
    --data_config config/data.yaml
```

#### Resume from checkpoint

```bash
python train.py \
    --training_config config/training.yaml \
    --data_config config/data.yaml \
    --resume_from_checkpoint checkpoints/checkpoint-5000
```

### 3. Evaluation

Evaluate the trained model on validation set:

```bash
python evaluate.py \
    --checkpoint checkpoints/best \
    --data_config config/data.yaml \
    --split val \
    --batch_size 8
```

Evaluate on training set:

```bash
python evaluate.py \
    --checkpoint checkpoints/best \
    --data_config config/data.yaml \
    --split train \
    --batch_size 8
```

### 4. Inference

#### Single audio file

```bash
python inference.py \
    --checkpoint checkpoints/best \
    --audio path/to/audio.wav
```

#### Batch inference on directory

```bash
python inference.py \
    --checkpoint checkpoints/best \
    --audio_dir path/to/audio/directory \
    --output transcriptions.txt \
    --batch_size 8
```

The output file will contain tab-separated values:
```
audio_00001.wav	تزک بابری از زہیر الدین محمد بابر
audio_00002.wav	پیش لفظ زیر نظر کتاب تزک بابری کا مصنف
```

## Training Details

### Model Architecture

- **Base Model**: `mahwizzzz/urd-wav2vec2-94m`
- **Task**: Connectionist Temporal Classification (CTC)
- **Language**: Urdu
- **Tokenizer**: Uses model's built-in processor

### Training Features

- **CTC Loss**: Connectionist Temporal Classification for sequence alignment
- **Mixed Precision (FP16)**: Faster training with lower memory usage
- **Gradient Accumulation**: Effective larger batch sizes
- **Automatic Mixed Precision**: Handled by HuggingFace Trainer
- **Best Model Saving**: Saves checkpoint with lowest validation WER
- **TensorBoard Logging**: Training metrics logged to `logs/` directory

### Hyperparameters

Default hyperparameters (configurable in `config/training.yaml`):

- Learning rate: 3.0e-4
- Batch size: 8 (per device)
- Gradient accumulation: 4 (effective batch size: 32)
- Epochs: 10
- Warmup steps: 500
- Weight decay: 0.01

### Memory Optimization

For GPUs with limited VRAM:

1. Reduce `per_device_train_batch_size` to 4 or 2
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_duration` in data config to limit sequence length

## Checkpoints

Checkpoints are saved in the `checkpoints/` directory:

- `checkpoint-{step}/`: Intermediate checkpoints
- `best/`: Best model based on validation WER
- `checkpoint-{step}/config.json`: Model configuration
- `checkpoint-{step}/preprocessor_config.json`: Processor configuration

Each checkpoint includes:
- Model weights (`pytorch_model.bin`)
- Model configuration
- Processor/tokenizer files

## Evaluation Metrics

The evaluation script computes:

- **WER (Word Error Rate)**: Primary metric for ASR
- **MER (Match Error Rate)**: Character-level accuracy
- **WIL (Word Information Lost)**: Information loss metric
- **WIP (Word Information Preserved)**: Information preservation metric

## GPU Requirements

### Minimum
- GPU: NVIDIA GPU with 8GB VRAM (e.g., GTX 1080)
- Batch size: 2-4
- Gradient accumulation: 8-16

### Recommended
- GPU: NVIDIA GPU with 16GB+ VRAM (e.g., Tesla T4, RTX 3090)
- Batch size: 8
- Gradient accumulation: 4

### Optimal
- GPU: NVIDIA GPU with 24GB+ VRAM (e.g., A100, RTX 4090)
- Batch size: 16
- Gradient accumulation: 2

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce batch size in `config/training.yaml`
2. Increase gradient accumulation steps
3. Reduce `max_duration` for audio files
4. Use gradient checkpointing (add to training config)

### Slow Training

1. Ensure FP16 is enabled (`fp16: true`)
2. Increase batch size if memory allows
3. Reduce `dataloader_num_workers` if CPU-bound
4. Check GPU utilization with `nvidia-smi`

### Audio Loading Errors

1. Verify all audio files exist
2. Check audio file formats (should be WAV)
3. Run `scripts/prepare_dataset.py` to validate dataset
4. Check file permissions

### Poor WER Results

1. Verify dataset quality and transcriptions
2. Increase training epochs
3. Adjust learning rate (try 1e-4 to 5e-4)
4. Check for class imbalance in dataset
5. Ensure sufficient training data (10k+ samples recommended)

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
├── inference.py             # Inference script
├── data/
│   ├── dataset.py           # Dataset class
│   └── preprocess.py        # Audio preprocessing
├── config/
│   ├── training.yaml        # Training configuration
│   └── data.yaml            # Data configuration
├── scripts/
│   ├── prepare_dataset.py   # Dataset validation
│   └── run_training.sh      # Training shell script
├── checkpoints/             # Model checkpoints (created during training)
├── logs/                    # TensorBoard logs (created during training)
└── metadata.csv             # Dataset metadata
```

## License

This repository is provided as-is for fine-tuning Urdu ASR models. The base model `mahwizzzz/urd-wav2vec2-94m` follows its original license.

## Citation

If you use this codebase, please cite:

```bibtex
@software{urdu_asr_wav2vec2,
  title = {Urdu ASR: Wav2Vec2 CTC Fine-tuning},
  author = {Mahwiz Khalil},
  year = {2024},
  url = {https://github.com/mwzkhalil/urd-wav2vec2-94m}
}
```

---


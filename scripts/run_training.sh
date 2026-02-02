#!/bin/bash

# Training script for Urdu ASR Wav2Vec2 fine-tuning
# This script sets up the environment and runs training

set -e

# Configuration
PYTHON_ENV="venv"
TRAINING_CONFIG="config/training.yaml"
DATA_CONFIG="config/data.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Urdu ASR Training Pipeline${NC}"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}CUDA available${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}Warning: CUDA not detected. Training will run on CPU (very slow).${NC}"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$PYTHON_ENV" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3.10 -m venv $PYTHON_ENV
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source $PYTHON_ENV/bin/activate

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Validate dataset
echo -e "${GREEN}Validating dataset...${NC}"
python scripts/prepare_dataset.py \
    --metadata metadata.csv \
    --audio_dir audio \
    --max_duration 30.0

# Run training
echo -e "${GREEN}Starting training...${NC}"
python train.py \
    --training_config $TRAINING_CONFIG \
    --data_config $DATA_CONFIG

echo -e "${GREEN}Training completed!${NC}"

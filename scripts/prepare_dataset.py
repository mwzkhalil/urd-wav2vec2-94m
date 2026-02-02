"""
Dataset preparation script.
Validates dataset structure and creates train/val splits.
"""

import sys
import argparse
import pandas as pd
import logging
from pathlib import Path

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocess import validate_audio_file, get_audio_duration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_dataset(metadata_path: str, audio_dir: str, max_duration: float = 30.0):
    """
    Validate dataset structure and files.
    
    Args:
        metadata_path: Path to metadata CSV
        audio_dir: Path to audio directory
        max_duration: Maximum audio duration
    """
    metadata_path = Path(metadata_path)
    audio_dir = Path(audio_dir)
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    logger.info(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    logger.info(f"Total samples in metadata: {len(df)}")
    
    required_columns = ['text', 'audio_path']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    valid_count = 0
    invalid_count = 0
    missing_files = 0
    invalid_duration = 0
    empty_text = 0
    
    for idx, row in df.iterrows():
        audio_path_str = str(row['audio_path'])
        
        if 'merged_dataset/audio/' in audio_path_str:
            audio_path = audio_dir / audio_path_str.replace('merged_dataset/audio/', '')
        else:
            audio_path = audio_dir / Path(audio_path_str).name
        
        if not audio_path.exists():
            missing_files += 1
            continue
        
        if not validate_audio_file(audio_path, max_duration):
            invalid_duration += 1
            continue
        
        text = str(row['text']).strip()
        if not text or len(text) < 1:
            empty_text += 1
            continue
        
        valid_count += 1
    
    invalid_count = missing_files + invalid_duration + empty_text
    
    logger.info("=" * 50)
    logger.info("Dataset Validation Results:")
    logger.info("=" * 50)
    logger.info(f"Valid samples: {valid_count}")
    logger.info(f"Invalid samples: {invalid_count}")
    logger.info(f"  - Missing files: {missing_files}")
    logger.info(f"  - Invalid duration: {invalid_duration}")
    logger.info(f"  - Empty text: {empty_text}")
    logger.info("=" * 50)
    
    return valid_count, invalid_count


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate Urdu ASR dataset")
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.csv",
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="audio",
        help="Path to audio directory"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds"
    )
    
    args = parser.parse_args()
    
    validate_dataset(args.metadata, args.audio_dir, args.max_duration)
    logger.info("Dataset validation completed!")


if __name__ == "__main__":
    main()

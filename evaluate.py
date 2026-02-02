"""
Evaluation script for Wav2Vec2 CTC model on Urdu ASR dataset.
Computes WER and other metrics on test/validation set.
"""

import argparse
import yaml
import logging
from pathlib import Path
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import jiwer
from tqdm import tqdm

from data.dataset import UrduASRDataset, collate_fn
from torch.utils.data import DataLoader


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_wer_metrics(predictions, references, processor):
    """
    Compute comprehensive WER metrics.
    """
    pred_str = processor.batch_decode(predictions)
    label_str = processor.batch_decode(references, group_tokens=False)
    
    wer = jiwer.wer(label_str, pred_str)
    mer = jiwer.mer(label_str, pred_str)
    wil = jiwer.wil(label_str, pred_str)
    wip = jiwer.wip(label_str, pred_str)
    
    return {
        "wer": wer,
        "mer": mer,
        "wil": wil,
        "wip": wip,
        "num_samples": len(pred_str)
    }


def evaluate_model(model, processor, dataset, batch_size=8, device="cuda"):
    """
    Evaluate model on dataset.
    """
    model.eval()
    model.to(device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        shuffle=False
    )
    
    all_predictions = []
    all_references = []
    
    logger.info(f"Evaluating on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)
            
            for i in range(predictions.shape[0]):
                pred = predictions[i].cpu().numpy()
                ref = labels[i].cpu().numpy()
                
                ref = ref[ref != -100]
                
                all_predictions.append(pred)
                all_references.append(ref)
    
    metrics = compute_wer_metrics(all_predictions, all_references, processor)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wav2Vec2 CTC model for Urdu ASR")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="config/data.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation"
    )
    
    args = parser.parse_args()
    
    data_config = load_config(args.data_config)
    
    logger.info(f"Loading model from {args.checkpoint}...")
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint)
    
    base_dir = Path.cwd()
    metadata_path = base_dir / data_config['dataset']['metadata_path']
    audio_dir = base_dir / data_config['dataset']['audio_dir']
    
    logger.info(f"Loading {args.split} dataset...")
    dataset = UrduASRDataset(
        metadata_path=str(metadata_path),
        audio_dir=str(audio_dir),
        processor=processor,
        max_duration=data_config['dataset']['max_duration'],
        min_duration=data_config['dataset']['min_duration'],
        split=args.split,
        train_ratio=data_config['dataset']['train_ratio'],
        seed=data_config['dataset']['seed']
    )
    
    logger.info("Running evaluation...")
    metrics = evaluate_model(
        model=model,
        processor=processor,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device
    )
    
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    logger.info(f"Word Error Rate (WER): {metrics['wer']:.4f}")
    logger.info(f"Match Error Rate (MER): {metrics['mer']:.4f}")
    logger.info(f"Word Information Lost (WIL): {metrics['wil']:.4f}")
    logger.info(f"Word Information Preserved (WIP): {metrics['wip']:.4f}")
    logger.info(f"Number of samples: {metrics['num_samples']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

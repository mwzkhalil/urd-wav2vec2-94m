"""
Training script for fine-tuning Wav2Vec2 CTC model for Urdu ASR.
Uses HuggingFace Trainer with CTC loss, mixed precision, and gradient accumulation.
"""

import yaml
import argparse
import logging
from pathlib import Path
import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
import numpy as np
import jiwer

from data.dataset import UrduASRDataset, collate_fn


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_wer(predictions, references, processor):
    """
    Compute Word Error Rate (WER) metric.
    """
    predictions = np.argmax(predictions, axis=-1)
    
    pred_str = processor.batch_decode(predictions)
    label_str = processor.batch_decode(references, group_tokens=False)
    
    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}


class CTCTrainer(Trainer):
    """
    Custom Trainer for CTC loss computation.
    """
    
    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute CTC loss with proper masking.
        """
        labels = inputs.get("labels")
        input_values = inputs.get("input_values")
        attention_mask = inputs.get("attention_mask")
        
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        logits = outputs.get("logits")
        
        loss = None
        if labels is not None:
            # CTC blank token is typically pad_token_id (usually 0 for Wav2Vec2)
            blank_token_id = model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else 0
            
            loss_fct = torch.nn.CTCLoss(blank=blank_token_id, reduction="mean", zero_infinity=True)
            
            input_lengths = attention_mask.sum(-1).long()
            label_lengths = (labels != -100).sum(-1).long()
            
            # Logits shape: (batch_size, seq_len, vocab_size)
            # CTC expects: (seq_len, batch_size, vocab_size)
            loss = loss_fct(
                logits.transpose(0, 1),
                labels,
                input_lengths,
                label_lengths
            )
        
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 CTC model for Urdu ASR")
    parser.add_argument(
        "--training_config",
        type=str,
        default="config/training.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="config/data.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    training_config = load_config(args.training_config)
    data_config = load_config(args.data_config)
    
    logger.info("Loading model and processor...")
    model_name = training_config['model']['model_name']
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    model.config.ctc_zero_infinity = True
    
    logger.info(f"Model loaded: {model_name}")
    logger.info(f"Vocab size: {len(processor.tokenizer)}")
    
    base_dir = Path.cwd()
    metadata_path = base_dir / data_config['dataset']['metadata_path']
    audio_dir = base_dir / data_config['dataset']['audio_dir']
    
    logger.info("Loading training dataset...")
    train_dataset = UrduASRDataset(
        metadata_path=str(metadata_path),
        audio_dir=str(audio_dir),
        processor=processor,
        max_duration=data_config['dataset']['max_duration'],
        min_duration=data_config['dataset']['min_duration'],
        split='train',
        train_ratio=data_config['dataset']['train_ratio'],
        seed=data_config['dataset']['seed']
    )
    
    logger.info("Loading validation dataset...")
    val_dataset = UrduASRDataset(
        metadata_path=str(metadata_path),
        audio_dir=str(audio_dir),
        processor=processor,
        max_duration=data_config['dataset']['max_duration'],
        min_duration=data_config['dataset']['min_duration'],
        split='val',
        train_ratio=data_config['dataset']['train_ratio'],
        seed=data_config['dataset']['seed']
    )
    
    training_args = TrainingArguments(
        output_dir=training_config['training']['output_dir'],
        logging_dir=training_config['training']['logging_dir'],
        num_train_epochs=training_config['training']['num_train_epochs'],
        per_device_train_batch_size=training_config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['training']['gradient_accumulation_steps'],
        learning_rate=training_config['training']['learning_rate'],
        warmup_steps=training_config['training']['warmup_steps'],
        weight_decay=training_config['training']['weight_decay'],
        fp16=training_config['training']['fp16'],
        dataloader_num_workers=training_config['training']['dataloader_num_workers'],
        save_strategy=training_config['training']['save_strategy'],
        evaluation_strategy=training_config['training']['evaluation_strategy'],
        save_total_limit=training_config['training']['save_total_limit'],
        load_best_model_at_end=training_config['training']['load_best_model_at_end'],
        metric_for_best_model=training_config['training']['metric_for_best_model'],
        greater_is_better=training_config['training']['greater_is_better'],
        logging_steps=training_config['training']['logging_steps'],
        eval_steps=training_config['training']['eval_steps'],
        save_steps=training_config['training']['save_steps'],
        seed=training_config['training']['seed'],
        report_to=training_config['training']['report_to'],
        push_to_hub=False,
    )
    
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        return compute_wer(predictions, labels, processor)
    
    trainer = CTCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        processor=processor,
    )
    
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(training_config['training']['output_dir'])
    
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss}")
    
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final WER: {eval_results.get('eval_wer', 'N/A')}")


if __name__ == "__main__":
    main()

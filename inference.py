"""
Inference script for Wav2Vec2 CTC model on Urdu ASR.
Transcribes audio files or audio streams.
"""

import argparse
import yaml
import logging
from pathlib import Path
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from data.preprocess import resample_audio, normalize_audio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def transcribe_audio(
    model,
    processor,
    audio_path: str,
    device: str = "cuda"
) -> str:
    """
    Transcribe a single audio file.
    
    Args:
        model: Wav2Vec2ForCTC model
        processor: Wav2Vec2Processor
        audio_path: Path to audio file
        device: Device to run inference on
        
    Returns:
        Transcribed text
    """
    model.eval()
    model.to(device)
    
    waveform = resample_audio(audio_path, target_sr=16000)
    waveform = normalize_audio(waveform)
    
    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        logits = outputs.logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription


def transcribe_batch(
    model,
    processor,
    audio_paths: list,
    device: str = "cuda",
    batch_size: int = 8
) -> list:
    """
    Transcribe multiple audio files in batches.
    
    Args:
        model: Wav2Vec2ForCTC model
        processor: Wav2Vec2Processor
        audio_paths: List of audio file paths
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        List of transcriptions
    """
    model.eval()
    model.to(device)
    
    transcriptions = []
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        batch_waveforms = []
        batch_attention_masks = []
        
        for audio_path in batch_paths:
            waveform = resample_audio(audio_path, target_sr=16000)
            waveform = normalize_audio(waveform)
            
            inputs = processor(
                waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            batch_waveforms.append(inputs.input_values.squeeze(0))
            batch_attention_masks.append(inputs.attention_mask.squeeze(0))
        
        max_len = max([w.shape[0] for w in batch_waveforms])
        
        padded_waveforms = []
        padded_masks = []
        
        for w, m in zip(batch_waveforms, batch_attention_masks):
            pad_len = max_len - w.shape[0]
            padded_w = torch.nn.functional.pad(w, (0, pad_len), value=0.0)
            padded_m = torch.nn.functional.pad(m, (0, pad_len), value=0)
            padded_waveforms.append(padded_w)
            padded_masks.append(padded_m)
        
        input_values = torch.stack(padded_waveforms).to(device)
        attention_mask = torch.stack(padded_masks).to(device)
        
        with torch.no_grad():
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        batch_transcriptions = processor.batch_decode(predicted_ids)
        
        transcriptions.extend(batch_transcriptions)
    
    return transcriptions


def main():
    parser = argparse.ArgumentParser(description="Inference with Wav2Vec2 CTC model for Urdu ASR")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to single audio file"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        help="Path to directory containing audio files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transcriptions.txt",
        help="Output file for transcriptions (when using --audio_dir)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for batch inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    if not args.audio and not args.audio_dir:
        parser.error("Either --audio or --audio_dir must be provided")
    
    logger.info(f"Loading model from {args.checkpoint}...")
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint)
    
    if args.audio:
        logger.info(f"Transcribing {args.audio}...")
        transcription = transcribe_audio(model, processor, args.audio, args.device)
        print(f"Transcription: {transcription}")
    
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
        audio_files.sort()
        
        logger.info(f"Found {len(audio_files)} audio files")
        logger.info("Transcribing...")
        
        transcriptions = transcribe_batch(
            model,
            processor,
            [str(f) for f in audio_files],
            args.device,
            args.batch_size
        )
        
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            for audio_file, transcription in zip(audio_files, transcriptions):
                f.write(f"{audio_file.name}\t{transcription}\n")
        
        logger.info(f"Transcriptions saved to {output_path}")


if __name__ == "__main__":
    main()

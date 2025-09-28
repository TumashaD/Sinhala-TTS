import os
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.utils.text import characters

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  # use ljspeech-style metadata format
    meta_file_train="phonemized.csv",
    path="dataset",
)

from TTS.tts.configs.tacotron2_config import Tacotron2Config

output_path = "output/tacotron2-DDC-sinhala"

# INITIALIZE THE TRAINING CONFIGURATION
config = Tacotron2Config()
config.load_json("output/tacotron2-DDC-sinhala/sinhala-ddc-September-13-2025_02+55AM-cbbc725/config.json")

# INITIALIZE THE AUDIO PROCESSOR
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

print(f"Training samples: {len(train_samples)}")
print(f"Evaluation samples: {len(eval_samples)}")

# INITIALIZE THE MODEL
model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

# INITIALIZE THE TRAINER
trainer = Trainer(
    TrainerArgs(
        continue_path="output/tacotron2-DDC-sinhala/sinhala-ddc-September-13-2025_02+55AM-cbbc725",
        gpu=0,
    ),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

# Print model info
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

if __name__ == "__main__":
    print("Starting Tacotron2 training for Sinhala...")
    # Start training
    trainer.fit()
    
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

SINHALA_IPA_CHARS = [
    # Basic vowels (matching the actual phonetic output)
    'a', 'aː', 'æ', 'æː', 'i', 'iː', 'u', 'uː', 'e', 'eː', 'o', 'oː',
    # Special vowels
    'ru', 'ruː', 'li', 'liː',
    # Additional vowels found in your data
    'ə', 'ɐ', 'ʊ', 'ɪ',  # schwa, near-open central, near-close near-back rounded, near-close near-front unrounded
    # Diphthongs
    'ai', 'au',
    
    # Consonants - stops
    'k', 'kʰ', 'g', 'gʰ', 'ŋ', 'ɡ',  # Added 'ɡ' (script g)
    't͡ʃ', 't͡ʃʰ', 'd͡ʒ', 'd͡ʒʰ', 'ɲ',
    'ʈ', 'ʈʰ', 'ɖ', 'ɖʰ', 'ɳ',
    't̪', 't̪ʰ', 'd̪', 'd̪ʰ', 'n̪',
    'p', 'pʰ', 'b', 'bʰ', 'm',
    
    # Sonorants
    'j', 'r', 'l', 'w', 'ɭ', 'ɹ',  # Added 'ɹ' (alveolar approximant)
    
    # Fricatives
    'ʃ', 'ʂ', 's', 'h', 'f', 'χ',  # Added 'χ' (voiceless uvular fricative)
    
    # Complex/compound consonants
    'ŋg', 'ɲd͡ʒ', 'ɳɖ', 'n̪d̪', 'mb',
    'gn',  # for ඥ
    
    # Prenasalized consonants (found in your data)
    'ᵐ', 'ⁿ', 'ᵑ',  # superscript m, n, ng for prenasalization
    
    # Special markers for compound sounds
    'X1', 'X2',  # markers used in the phonetic rules
    
    # Numbers
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]



characters_string = "".join(SINHALA_IPA_CHARS)
# Include ” and ‘ in punctuation"
punctuation_string = ".,!?;:=()-[]\"'“”‘’෴ \n\r\t#"

output_path = "output/tacotron2-sinhala"

# INITIALIZE THE TRAINING CONFIGURATION
config = Tacotron2Config()
config.load_json("C:\\Users\\tumas\\OneDrive\\Desktop\\Sinhala-TTS\\output\\tacotron2-sinhala\\run-September-10-2025_12+43AM-4e96c01\\config.json")

print(config.test_sentences)

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
        continue_path="C:\\Users\\tumas\\OneDrive\\Desktop\\Sinhala-TTS\\output\\tacotron2-sinhala\\run-September-10-2025_12+43AM-4e96c01",  # Set this to resume from checkpoint
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
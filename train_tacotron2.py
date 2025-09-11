import os
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import CharactersConfig

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
config = Tacotron2Config(
    batch_size=32,
    eval_batch_size=16,
    mixed_precision=True,
    allow_tf32=True, 
    use_grad_scaler=True,
    run_eval=True,
    test_delay_epochs=10,
    epochs=1000, 
    num_loader_workers=6,
    num_eval_loader_workers=3,

    optimizer="RAdam",  # Use RAdam like reference
    optimizer_params={
        "betas": [0.9, 0.998],  # Match reference
        "weight_decay": 1e-6,
    },

    # Add gradual training like the reference
    gradual_training=[
        [0, 6, 32],      # Start with r=6, batch_size=32
        [10000, 4, 32],  # After 10k steps: r=4
        [50000, 3, 32],  # After 50k steps: r=3  
        [100000, 2, 32], # After 100k steps: r=2
    ],

    r=6,  # Start with higher r value
    max_decoder_steps=2000,

    # Use the loss weights from reference config
    decoder_loss_alpha=0.25,
    postnet_loss_alpha=0.25,
    decoder_diff_spec_alpha=0.25,  # Enable diff spec loss
    postnet_diff_spec_alpha=0.25,  # Enable diff spec loss
    decoder_ssim_alpha=0.25,       # Enable SSIM loss
    postnet_ssim_alpha=0.25,       # Enable SSIM loss
    ga_alpha=5.0,
    stopnet_pos_weight=15.0,

    # Characters
    characters= CharactersConfig(
        pad="<PAD>",      # Padding token
        eos="<EOS>",      # End of sequence
        bos="<BOS>",      # Beginning of sequence
        blank="<BLANK>",  # Blank token for CTC-based models
        characters=characters_string,
        punctuations=punctuation_string,
        is_unique=True,   # Remove duplicates
        is_sorted=False,  # Keep our logical ordering
    ),

    # Audio
    audio={
        # Basic audio settings
        "sample_rate": 22050,  # Standard TTS sample rate
        "resample": True,      # Enable resampling if needed
        "do_trim_silence": True,
        "trim_db": 45,        # Trim silence threshold

        # Sound normalization
        "do_sound_norm": True, # Enable sound normalization
        "db_level": -25,      # Target dB level for normalization
        "do_rms_norm": True,  # Enable RMS normalization

        # Spectrogram settings
        "num_mels": 80,       # Number of mel bands
        "fft_size": 1024,     # FFT window size
        "hop_length": 256,    # Number of samples between frames
        "win_length": 1024,   # Window length
        "mel_fmin": 0,        # Minimum mel frequency
        "mel_fmax": 8000,     # Maximum mel frequency

        # Normalization settings
        "signal_norm": True,
        "symmetric_norm": True,
        "max_norm": 1.0,      # Normalization range
        "clip_norm": True,    # Clip values outside range

        # Other processing
        "preemphasis": 0.97,  # Pre-emphasis coefficient
        "ref_level_db": 20,   # Reference level dB
        "min_level_db": -100, # Minimum level dB
        "power": 1.5,        # Power for spectral normalization
        "griffin_lim_iters": 60
    },

    test_sentences=[
        "aːdaːnə mewələm utsaːhə kɐɹənnə.",
        "pɐhətə dækwenə obeː bʰaːʃaːw sɐhə aːdaːnə mewələm toːɹaː ʈɐjip kiɹiːmə ɐɹəᵐbənnə.",
        "obə sitənneː kuməkdæji ɐpəʈə dɐnwənnə."
    ],

    lr=2e-4, 
    lr_scheduler="ExponentialLR",
    lr_scheduler_params={
        "gamma": 0.999,  # Decay factor per step
        "last_epoch": -1
    },

    grad_clip=1.0, 


    # Logging
    print_step=20,
    save_step=500,

    # Output
    output_path=output_path,
    datasets=[dataset_config],

    # Evaluation
    eval_split_size=0.1,
    eval_split_max_size=100,
)

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
        grad_accum_steps=2,
        
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
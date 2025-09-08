import os
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import CharactersConfig
import torch

# GPU optimizations for RTX 3050
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# CPU optimizations
os.environ["OMP_NUM_THREADS"] = "8" 
os.environ["MKL_NUM_THREADS"] = "8"

# Memory management
torch.cuda.empty_cache()

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  # use ljspeech-style metadata format
    meta_file_train="metadata.csv",
    path="dataset"
)

from TTS.tts.configs.tacotron2_config import Tacotron2Config

SINHALA_IPA_CHARS = [
    # Basic vowels
    'a', 'aː', 'æ', 'æː', 'i', 'iː', 'u', 'uː', 'e', 'eː', 'o', 'oː', 'ə',
    # Diphthongs and special vowels
    'ai', 'au', 'ri', 'ru', 'ruː',
    # Consonants (including complex ones)
    'k', 'g', 'ŋ', 'c', 'j', 'ɲ', 'ʈ', 'ɖ', 'n', 'p', 'b', 'm', 'r', 'l', 'v', 'f',
    'ʃ', 'ʂ', 's', 'h', 'ɭ',
    # Dental consonants
    't̪', 'd̪',
    # Complex consonants
    'dʒ', 'ŋɡ', 'dʒɲ', 'ɖn', 'nd̪', 'mb',
    # Numbers
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]

characters_string = "".join(SINHALA_IPA_CHARS)
# Include ” and ‘ in punctuation"
punctuation_string = ".,!?;:=()-[]\"'“”‘’ \n\r\t" 

output_path = "output/tacotron2-sinhala"

# INITIALIZE THE TRAINING CONFIGURATION
config = Tacotron2Config(
    batch_size=16,
    eval_batch_size=8,

    mixed_precision=True,
    use_grad_scaler=True,
    allow_tf32=True,
    cudnn_benchmark=True,

    num_loader_workers=8,
    num_eval_loader_workers=4,

    run_eval=True,
    test_delay_epochs=5,
    epochs=1000, 

    out_channels=80,  # Number of mel channels

    # Model parameters
    r=2,  
    max_decoder_steps=2000,

    # Loss weights
    decoder_loss_alpha=0.25,
    postnet_loss_alpha=0.25,
    decoder_diff_spec_alpha=0.25,
    postnet_diff_spec_alpha=0.25,
    decoder_ssim_alpha=0.25,
    postnet_ssim_alpha=0.25,
    ga_alpha=5.0,

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
        "do_rms_norm": True,   # Enable RMS normalization
        "db_level": -27,      # Target dB level for normalization

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
        "max_norm": 4.0,      # Normalization range
        "clip_norm": True,    # Clip values outside range

        # Other processing
        "preemphasis": 0.97,  # Pre-emphasis coefficient
        "ref_level_db": 20,   # Reference level dB
        "min_level_db": -100, # Minimum level dB
        "power": 1.5,        # Power for spectral normalization
        "griffin_lim_iters": 60
    },

    test_sentences=[
        "ad̪ə pason pahojo d̪inəjəji. pason pahojo apə raʈəʈə at̪iʃəjə væd̪əgət̪ d̪inəjək.",
        "ad̪ə mamə kaʈuvəkin vid̪unaː lad̪d̪ə jamsə d̪uk vind̪im d̪ə, mase mə panvaːlə radʒət̪əme jud̪d̪əjəhi d̪iː iːjan vid̪inə lad̪d̪ə d̪uk vind̪iːvaː.",
        "ad̪arməkarməjəhi ad̪arməkarməsaŋɲaː æt̪t̪iː nam d̪ukuɭaː ævæt̪ va."
    ],

    # Training
    lr=1e-4,

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
        continue_path="",  # Set this to resume from checkpoint
        restore_path="",   # Set this to fine-tune from pretrained model
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



import os
from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import hifigan_config
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN
from TTS.config.shared_configs import BaseAudioConfig

output_path = "output/hifigan_Dinithi"

config = hifigan_config.HifiganConfig()
config.load_json("C:\\Users\\tumas\\AppData\\Local\\tts\\vocoder_models--en--ljspeech--hifigan_v2\\finetune_config.json")

ap = AudioProcessor(**config.audio)
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

model = GAN(config,ap)

trainer = Trainer(
    TrainerArgs(
        continue_path="output/hifigan_Dinithi/hifigan-October-07-2025_05+18PM-c742b21",
        gpu=0,
    ),
    config=config,
    output_path=output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)


print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

if __name__ == "__main__":
    print("Starting HiFi-GAN training for Sinhala...")
    trainer.fit()
from TTS.utils.synthesizer import Synthesizer

model_path = "sinhala.pth"
config_path = "config.json"

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
)

wav = synthesizer.tts("api tamā chehān, dilhāra saha ṭumāsha")
synthesizer.save_wav(wav, "output.wav")
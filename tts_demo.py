from TTS.utils.synthesizer import Synthesizer

tts_model_path = "C:\\Users\\tumas\\OneDrive\\Desktop\\Sinhala-TTS\\output\\tacotron2-sinhala\\run-September-10-2025_12+43AM-4e96c01\\best_model.pth"
tts_config_path = "C:\\Users\\tumas\\OneDrive\\Desktop\\Sinhala-TTS\\output\\tacotron2-sinhala\\run-September-10-2025_12+43AM-4e96c01\\config.json"
vocoder_model_path = "C:\\Users\\tumas\\AppData\\Local\\tts\\vocoder_models--universal--libri-tts--wavegrad\\model_file.pth"
vocoder_config_path = "C:\\Users\\tumas\\AppData\\Local\\tts\\vocoder_models--universal--libri-tts--wavegrad\\config.json"

synthesizer = Synthesizer(
    tts_checkpoint=tts_model_path,
    tts_config_path=tts_config_path,
    vocoder_checkpoint=vocoder_model_path,
    vocoder_config=vocoder_config_path,
)

wav = synthesizer.tts("pɐhətə dækwenə obeː bʰaːʃaːw sɐhə aːdaːnə mewələm toːɹaː ʈɐjip kiɹiːmə ɐɹəᵐbənnə.")
synthesizer.save_wav(wav, "sinhala2.wav")
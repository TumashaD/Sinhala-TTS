from TTS.utils.synthesizer import Synthesizer
from g2p import convert_text
from TTS.api import TTS

tts_model_path = "models\\dinithi.pth"
tts_config_path = "models\\dinithi.json"
vocoder_model_path = "models\\dinithi_vocoder.pth"
vocoder_config_path = "models\\dinithi_vocoder.json"

synthesizer = Synthesizer(
    tts_checkpoint=tts_model_path,
    tts_config_path=tts_config_path,
    vocoder_checkpoint=vocoder_model_path,
    vocoder_config=vocoder_config_path,
)

text = "මියුරු එක්කනම් මේක කරන්න බැහැ."
wav = synthesizer.tts(convert_text(text))
synthesizer.save_wav(wav, "output.wav")
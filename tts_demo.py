from ctypes import alignment
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models import tacotron2
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
import torch
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.synthesis import synthesis
from TTS.utils import generic_utils
from TTS.tts.models import setup_model
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.config import load_config
import IPython
from IPython.display import Audio
import os
import numpy as np
import json

tts_model_path = "output\\tacotron2-DDC-sinhala\\sinhala-ddc-September-13-2025_02+55AM-cbbc725\\checkpoint_303000.pth"
tts_config_path = "output\\tacotron2-DDC-sinhala\\sinhala-ddc-September-13-2025_02+55AM-cbbc725\\config.json"
vocoder_model_path = "C:\\Users\\tumas\\AppData\\Local\\tts\\vocoder_models--en--sam--hifigan_v2\\model_file.pth"
vocoder_config_path = "C:\\Users\\tumas\\AppData\\Local\\tts\\vocoder_models--en--sam--hifigan_v2\\config.json"

synthesizer = Synthesizer(
    tts_checkpoint=tts_model_path,
    tts_config_path=tts_config_path,
    vocoder_checkpoint=vocoder_model_path,
    vocoder_config=vocoder_config_path,
)
text = "ɡowi dʒɐnətaːwəɡeː, dʰiːwəɹə dʒɐnətaːwəɡeː, wɐtu kɐmkəɹuwaːɡeː meː sijəlu ɡ ɹaːmiːjə dʒɐnətaːwəɡeː ɡæʈəlu sɐməɡə kɐʈəjutu kɐɹənəwaː."
wav = synthesizer.tts(text)
synthesizer.save_wav(wav, "test.wav")

# config = load_config(tts_config_path)
# model = tacotron2.Tacotron2.init_from_config(config)
# ap = AudioProcessor(**config.audio)
# cp = torch.load(tts_model_path, map_location=torch.device('cpu'))
# model.load_state_dict(cp['model'])
# model.eval()
# tokenizer, config = TTSTokenizer.init_from_config(config)
# text = "pɐhətə dækwenə obeː bʰaːʃaːw sɐhə aːdaːnə mewələm toːɹaː ʈɐjip kiɹiːmə ɐɹəᵐbənnə."
# input_text = tokenizer.text_to_ids(text)
# input_text = torch.LongTensor(input_text).unsqueeze(0)

# output = synthesis(
#         model=model,
#         text=text,
#         CONFIG=config,
#         use_cuda=False,
#         use_griffin_lim=True,
#         do_trim_silence=False,
#     )
# alignments = output["outputs"]["alignments"]
# postnet_outputs = output["outputs"]["model_outputs"]
# fig = plot_alignment(alignments, fig_size=(8, 5))
# fig.savefig("alignment.png")
# fig = plot_spectrogram(postnet_outputs, ap, fig_size=(10, 5), output_fig=True)
# fig.savefig("postnet_spectrogram.png")

# # Generate the waveform from the mel spectrogram using external vocoder

# with open(vocoder_config_path, 'r') as f:
#     vocoder_config = json.load(f)

# vocoder = torch.jit.load(vocoder_model_path, map_location=torch.device('cpu'))
# vocoder.eval()

# with torch.no_grad():
#     mel = torch.FloatTensor(postnet_outputs).unsqueeze(0)
#     audio = vocoder(mel)
#     audio = audio.squeeze().cpu().numpy()
    
# # Save using AudioProcessor from TTS
# ap.save_wav(audio, "testing.wav")
import pickle
import os
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment, notebook
import scipy.io.wavfile as wavfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transcript_utils import plot_transcript, TranscriptLoader
from audio_utils import AudioLoader

plot_with_transcript = True

data_path = "//talebase//data//speech_raw//switchboard_1"

# Hyper-parameters
sr = 16000

# Audio parameters
conversation_id = "2032"
audio_duration = 10 # seconds
start = 25 # seconds
max_len = 500
end = start + audio_duration # seconds

audio_loader = AudioLoader(
    [os.path.join(data_path, "swb1_d{0}//data".format(i)) for i in range(1, 5)],
    sr=sr
)

transcript_loader = TranscriptLoader(os.path.join(data_path, "swb_ms98_transcriptions"))

# Get audio
audio_input = audio_loader.load_audiofile("sw0{0}.sph".format(conversation_id))[:,int(start*sr):int(end*sr)]
print("audio_input", audio_input.shape)

os.makedirs("temp_files//{0}".format(conversation_id), exist_ok=True)
audio_A ="temp_files//{0}//orignal_audio_A.wav".format(conversation_id)
audio_B ="temp_files//{0}//orignal_audio_B.wav".format(conversation_id)
wavfile.write(audio_A, sr, audio_input[0])
wavfile.write(audio_B, sr, audio_input[1])

# Load transcripts
transcriptA = transcript_loader.get_transcript(conversation_id, "A")
transcriptB = transcript_loader.get_transcript(conversation_id, "B")

# Load vad
model = Model.from_pretrained("segmentation//pytorch_model.bin", 
                              use_auth_token="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")

pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)

# Find turns
vad_A = pipeline(audio_A)
vad_B = pipeline(audio_B)

vad_A = vad_A.rename_labels({label: "A" for label in vad_A.labels()})
vad_B = vad_B.rename_labels({label: "B" for label in vad_B.labels()})

vad_A = vad_A.rename_tracks("int")
vad_B = vad_B.rename_tracks("int")

notebook.crop = Segment(0, end-start)

# Plot turns
fig, axs = plt.subplots(2, 1, figsize=(6, 4), constrained_layout=True)
notebook.plot_annotation(vad_A, ax=axs[0])
notebook.plot_annotation(vad_B, ax=axs[1])

if plot_with_transcript: # Plot transcripts
    plot_transcript(axs[0], transcriptA, start=start, end=end)
    plot_transcript(axs[1], transcriptB, start=start, end=end)

axs[0].set_title("Speaker A")
axs[1].set_title("Speaker B")
axs[0].set_xlabel("Time (s)")
axs[1].set_xlabel("Time (s)")

plt.savefig(os.path.join("results", "plots", "transcript_vad_alignment.png"), dpi=300)

print("vad_A", vad_A)
print("vad_B", vad_B)

print("transcriptA")
[print(element.content) for element in transcriptA.get_elements_between(start, end, include_edge=True)]

print("transcriptB")
[print(element.content) for element in transcriptB.get_elements_between(start, end, include_edge=True)]


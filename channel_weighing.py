import pickle
import torch
import os
import numpy as np

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment, notebook

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from transcript_utils import label_x_with_transcript, plot_transcript

test_name = "4023"
channel = 0
vad = True
use_seconds = False # use seconds instead of units
line = False 

generated_audio_file_A = os.path.join("results", test_name, "generated_audio_A.wav")
generated_audio_file_B = os.path.join("results", test_name, "generated_audio_B.wav")
file_name = os.path.join("results", test_name, "result_container.pkl")

with open(file_name, "rb") as f:
    result_container = pickle.load(f)

generated_sequences = result_container.generated_sequences
transcriptA = result_container.transcripts[0]
transcriptB = result_container.transcripts[1]
start = result_container.start
end = result_container.end
src_length = result_container.src_length
max_len = result_container.tgt_length
rel_fs = result_container.rel_fs

gradients = generated_sequences[0]["input_gradients"]
saliency = [[], []]
for channel, channel_gradients in enumerate(gradients):
    saliency[channel] = torch.stack(channel_gradients, dim=0).squeeze().abs() # (generated_units, channels, max_len)
    saliency[channel] = saliency[channel].transpose(0, 2) / saliency[channel].sum(dim=(1,2)) # Normalize
    saliency[channel] = saliency[channel].transpose(0, 2) # Transpose back

saliency = torch.stack(saliency).transpose(1, 2) # (saliency_channels, channels, generated_units, max_len)
saliency_per_channel = saliency.sum(dim=3).detach().cpu().numpy() # (saliency_channels, channels, generated_units)


if vad:
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
    vad_A = pipeline(generated_audio_file_A)
    vad_B = pipeline(generated_audio_file_B)

    vad_A = vad_A.rename_labels({label: "Speaker A" for label in vad_A.labels()})
    vad_B = vad_B.rename_labels({label: "Speaker B" for label in vad_B.labels()})

    vad_both = vad_B.update(vad_A, copy=True).crop(Segment(0, max_len * rel_fs))
    notebook.crop = Segment(src_length * rel_fs, max_len * rel_fs)

##################### Plotting #####################

print("Plotting heatmaps...")

plots = 2
num_axes = plots + 1 if vad else plots
fig, axs = plt.subplots(num_axes, 1, figsize=(8.5, 5), constrained_layout=True)

if vad:
    notebook.plot_annotation(vad_both, ax=axs[2])

cm = matplotlib.cm.get_cmap("Set1")
colors = [cm(1.0 * i / 8) for i in range(9)]
_, lbls = axs[2].get_legend_handles_labels()
if lbls[0] == "Speaker A":
    print(lbls[0])
    speakerA_color = colors[0] #"blue"
    speakerB_color = colors[1] #"red"
elif lbls[0] == "Speaker B":
    print(lbls[0])
    speakerA_color = colors[1] # "red"
    speakerB_color = colors[0] # "blue"

for channel in range(2):
    axs[channel].plot(saliency_per_channel[channel, 0], color=speakerA_color, label="Speaker A")
    axs[channel].plot(saliency_per_channel[channel, 1], color=speakerB_color, label="Speaker B")
    axs[channel].legend()

print("Plotted heatmaps")

if line:
    linewidth = 0.3

    x = np.arange(src_length * 2, max_len * 2) // 2 + 1 + linewidth + 0.7
    y = np.arange(0, (max_len-src_length + 1) * 2) // 2 - linewidth
    #x = x[1:-1]
    y = y[1:-1]
    
    print(x)
    print(y)
    axs[0].plot(x, y, color="red", linewidth=linewidth)

##################### Formatting #####################

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=12)

if use_seconds: # Format ticks as seconds
    # xticks
    num_xticks = len(axs[0].get_xticks())
    xtick_pos = np.linspace(0, max_len, num_xticks)
    xtick_labels = np.round(np.linspace(0, max_len * rel_fs, num_xticks), decimals=1)
    axs[0].set_xticks(xtick_pos, xtick_labels)

    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Generated time")

    # yticks
    num_yticks = len(axs[0].get_yticks())
    generated_len = max_len-src_length
    ytick_pos = np.linspace(0, generated_len, num_yticks)
    ytick_labels = np.round(np.linspace(src_length * rel_fs, (src_length + generated_len) * rel_fs, num_yticks), decimals=1)
    axs[0].set_yticks(ytick_pos, ytick_labels, fontsize=16)
else: # Format ticks as units
    axs[2].set_xlabel("Generated units", fontsize=14)
    
    axs[0].set_ylabel("Weighing", fontsize=14)
    axs[1].set_ylabel("Weighing", fontsize=14)

# Make sure xticks are the same for both heatmaps
axs[1].set_xticks(axs[0].get_xticks(), axs[0].get_xticklabels()) # xticks
axs[1].set_xlim(axs[0].get_xlim())

for ax in axs[:2]:
    ax.set_ylim(0, 1)


if vad:
    saliency_xticks = axs[0].get_xticks()
    saliency_xticklabels = axs[0].get_xticklabels()
    tick_pos = np.asarray(saliency_xticks) * rel_fs + src_length * rel_fs
    #tick_pos = np.linspace(0, (max_len - src_length) * rel_fs, len(saliency_xticklabels))
    print("tick_pos", tick_pos)
    print("heat_map_xlabels", saliency_xticklabels)
    axs[2].set_xticks(tick_pos, saliency_xticklabels)

    axs[2].set_xlim(np.asarray(axs[0].get_xlim()) * rel_fs + src_length * rel_fs)


# Titles
axs[0].set_title("Saliency from speaker A", fontsize=16)
axs[1].set_title("Saliency from speaker B", fontsize=16)
if vad:
    axs[2].set_title("Turn segmentation", fontsize=16)

# Seconds formatting
if False:
    formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: str(x) + "s")
    for ax in axs:
        ax.xaxis.set_major_formatter(formatter)

if vad:
    for ax in axs[:2]:
        ax.set_xticks(ax.get_xticks(), ['' for x in ax.get_xticks()])
        ax.set_xlim(np.asarray(axs[2].get_xlim()) / rel_fs - src_length)

print("Saving plot...")
fig.savefig("results//plots//channel_weighing.png", dpi=300)
fig.savefig(os.path.join("results", test_name, test_name+"_channel_weighing"), dpi=300)
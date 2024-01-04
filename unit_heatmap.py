import pickle
import torch
import os
import numpy as np
import time

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment, notebook

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from transcript_utils import label_x_with_transcript, plot_transcript

# Hyper-parameters
test_name = "4024" # Name of folder containing results
channel = 0
word_ticks = True
vad = True
use_seconds = False # NOTE: not supported anymore
line = False # Plot diagonal line showing where the input ends

# Get file paths
generated_audio_file_A = os.path.join("results", test_name, "generated_audio_A.wav")
generated_audio_file_B = os.path.join("results", test_name, "generated_audio_B.wav")
file_name = os.path.join("results", test_name, "result_container.pkl")

# Load results
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

# Load gradients
gradients = torch.stack(generated_sequences[0]["input_gradients"][channel], dim=0).squeeze() # dim: (generated_units, channels, max_len)

# Calculate saliency map
saliency_map = gradients.abs()

# Normalize saliency map
saliency_map_sum = saliency_map.sum(dim=(1,2))
saliency_map_sum[saliency_map_sum == 0] = 1 # Incase of zero gradients
saliency_map = saliency_map.transpose(0, 2) / saliency_map_sum # Normalize
saliency_map = saliency_map.transpose(0, 2) # Transpose back

# Extract saliency map for each channel
channel_A_saliency_map = saliency_map[:,0,:].detach().cpu().numpy()
channel_B_saliency_map = saliency_map[:,1,:].detach().cpu().numpy()


if vad: # Calulate turn segmentation
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

    vad_both = vad_B.update(vad_A, copy=True)
    notebook.crop = Segment(0, max_len * rel_fs)

##################### Plotting #####################

print("Plotting heatmaps...")

num_axes = 3 if vad else 2
fig, axs = plt.subplots(num_axes, 1, figsize=(10, 12), constrained_layout=True)

# Find max value for colorbar
vmax = np.max([np.max(channel_A_saliency_map), np.max(channel_B_saliency_map)])

# Plot heatmaps
g1 = sns.heatmap(channel_A_saliency_map, vmin=0, vmax=vmax, ax=axs[0], cmap="viridis")
g2 = sns.heatmap(channel_B_saliency_map, vmin=0, vmax=vmax, ax=axs[1], cmap="viridis")

print("Plotted heatmaps")

if vad: # Plot turn segmentation
    notebook.plot_annotation(vad_both, ax=axs[2])

if line:
    for ax in axs[:2] if vad else axs:
        linewidth = 0.3

        x = np.arange(src_length * 2, max_len * 2) // 2 + 1 + linewidth + 0.7
        y = np.arange(0, (max_len-src_length + 1) * 2) // 2 - linewidth
        #x = x[1:-1]
        y = y[1:-1]
        
        ax.plot(x, y, color="red", linewidth=linewidth)

##################### Formatting #####################

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=12)

if use_seconds: # Format ticks as seconds. NOTE: Not supported anymore
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
    for ax in axs:
        ax.set_xlabel("Input units", fontsize=14)
    
    axs[0].set_ylabel("Generated units", fontsize=14)
    axs[1].set_ylabel("Generated units", fontsize=14)

# Make sure yticks are the same for both heatmaps
axs[1].set_yticks(axs[0].get_yticks(), axs[0].get_yticklabels()) # yticks
axs[1].set_xticks(axs[0].get_xticks(), axs[0].get_xticklabels()) # xticks
if vad:
    saliency_xticks = axs[0].get_xticks()
    saliency_xticklabels = axs[0].get_xticklabels()
    tick_pos = np.asarray(saliency_xticks) * rel_fs
    axs[2].set_xticks(tick_pos, saliency_xticklabels, rotation=90)
    axs[2].set_xlim(np.asarray(axs[0].get_xlim()) * rel_fs)

# Plot transcripts
if word_ticks:
    label_x_with_transcript(axs[0], transcriptA, start, end, fontsize=9, pos="left", rel_fs=rel_fs, rotation=90, upper=True)
    label_x_with_transcript(axs[1], transcriptB, start, end, fontsize=9, pos="left", rel_fs=rel_fs, rotation=90, upper=True)

# Titles
axs[0].set_title("Speaker A*", fontsize=16)
axs[1].set_title("Speaker B", fontsize=16)
if vad:
    axs[2].set_title("Turn segmentation", fontsize=16)


#fig.suptitle("Unit heatmap")
axs[2].annotate('* = Saliency calculated with respect to this speaker',
            xy = (1.0, -0.2),
            xycoords='axes fraction',
            ha='right',
            va="center",
            fontsize=9)

print("Saving plot...")
fig.savefig("results//plots//unit_heatmap.png", dpi=300)
fig.savefig(os.path.join("results", test_name, test_name+"unit_heatmap"), dpi=300)
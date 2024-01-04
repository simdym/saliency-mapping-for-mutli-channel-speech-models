import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from examples.textless_nlp.dgslm.dgslm_utils import ApplyKmeans
from fairseq.checkpoint_utils import load_model_ensemble

from audio_utils import AudioLoader
from transcript_utils import load_isip_transcript, plot_transcript


models, _ = load_model_ensemble(["model_files//hubert_fisher.pt"])
hubert = models[0]

quantizer = ApplyKmeans("model_files//hubert_fisher_km_500.bin")

# Move model to GPU
hubert.cuda()

# Disable dropout
hubert.eval()

# Data loading
data_path = "//talebase//data//speech_raw//switchboard_1"
sr = 16000

# Rel. frequency scaling for transcript
stride = 320 # samples
receptive_field = 400
rel_fs = stride / sr

audio_loader = AudioLoader(
    [os.path.join(data_path, "swb1_d{0}//data".format(i)) for i in range(1, 5)],
    sr=sr
)

start = 80
end = start + 5
channel = 0
plotting = True

stds = []
means = []
midpoints = []

n = 5
m = 50 # SmoothGrad repetitions
noise_var = 0.01

np.random.seed(seed=100)
torch.manual_seed(100)
for i, target_unit_idx in enumerate(np.random.randint(0, 249, n)):
    total_saliency_map = np.zeros(int((end-start) * sr))
    for j in range(m):
        input = torch.Tensor(audio_loader.load_audiofile("sw02032.sph"))[:,int(start*sr):int(end*sr)].cuda()

        if m > 1:
            input += torch.randn_like(input) * noise_var
        print("input", input.size())
        input.requires_grad_()

        feats, _ = hubert.extract_features(
            source=input,
            padding_mask=None,
            mask=False,
            output_layer=12 # 12
        )
        
        featA = feats[channel] # Only channel 0 or A

        quantized_units_A, dist = quantizer(featA)

        unit = quantized_units_A[target_unit_idx]
        score = dist[target_unit_idx][unit]
        print("score", score)
        score.backward()

        saliency_map = input.grad[0].abs().cpu().numpy()
        saliency_map = saliency_map / np.sum(saliency_map)

        total_saliency_map += saliency_map

        hubert.zero_grad()
        input.grad.zero_()
    
    saliency_map = total_saliency_map / m

    frame_start = (target_unit_idx) * rel_fs
    frame_end = frame_start + receptive_field / sr
    frame_midpoint = (frame_end + frame_start) / 2

    time = np.linspace(0, end-start, len(saliency_map))
    saliency_map_time_mean = np.average(time, weights=saliency_map)
    print("target_unit_idx", target_unit_idx)
    print("target_unit_idx * rel_fs", target_unit_idx * rel_fs)
    print("midpoint", frame_midpoint)
    print("saliency_map_time_mean", saliency_map_time_mean)

    saliency_map_time_var= np.sum(saliency_map * (time - saliency_map_time_mean)**2)
    saliency_map_time_std = np.sqrt(saliency_map_time_var)

    np.std(saliency_map * np.linspace(start, end, len(saliency_map)))
    print("std_saliency_map", saliency_map_time_std)
    stds.append(saliency_map_time_std)
    means.append(saliency_map_time_mean)
    midpoints.append(frame_midpoint)


    if plotting:
        print("Plotting")
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

        ax1.plot(np.linspace(0, end-start, len(saliency_map)), saliency_map, linewidth=0.7)
        print("frame_start", frame_start)
        print("frame_end", frame_end)
        print(ax1.get_ylim())

        ylim = ax1.get_ylim()
        ax1.fill_betweenx(ylim, frame_start, frame_end, facecolor="lightcoral", label="Feature encoder\n receptive field")
        ax1.axvline(x=frame_start, color="red", linewidth=0.3)
        ax1.axvline(x=frame_end, color="red", linewidth=0.3)


        ax1.set_ylabel("Saliency", fontsize=12)
        ax1.set_xlabel("Time (s)", fontsize=12)
        ax1.set_ylim(ylim)

        ax1.legend()
        fig.tight_layout()
    
        print("Saving figure")
        fig.savefig("results//plots//hubert_saliency_map//{0}_hubert_sm_{1}.png".format(target_unit_idx, i), dpi=300)
    print("Done")

print("stds", stds)
print("mean_std", np.mean(stds))
print("means", means)
print("midpoints", midpoints)
print("mean_bias", np.abs(np.asarray(midpoints) - np.asarray(means)))
print("mean bias", np.mean(np.abs(np.asarray(midpoints) - np.asarray(means))))
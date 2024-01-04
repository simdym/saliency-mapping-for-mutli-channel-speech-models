import fairseq
import os
import torch
import numpy as np
import logging
import pickle
import json
import scipy.io.wavfile as wavfile
import shutil

from saliency_mapping import SaliencyResult
from audio_utils import AudioLoader
from transcript_utils import load_isip_transcript, plot_transcript, label_x_with_transcript, TranscriptLoader

from fairseq.models.speech_dlm import SpeechDLM
from examples.textless_nlp.dgslm.dgslm_utils import HifiganVocoder, ApplyKmeans

torch.manual_seed(0)

print("Started")

# Set logging level
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Save paths to models and data
hubert_path= "model_files//hubert_fisher.pt"
kmeans_path = "model_files//hubert_fisher_km_500.bin"
vocoder_path = "model_files//hifigan_vocoder"
vocoder_cfg_path = "model_files//config.json"
data_path = "//talebase//data//speech_raw//switchboard_1"

# Hyper-parameters 
sr = 16000
hubert_layer = 12

# Audio parameters
conversation_id = "4024" # 83
audio_duration = 5 # seconds
start = 25 # seconds
max_len = 500
end = start + audio_duration # seconds

# Saving parameter
run_title = conversation_id

# Rel. frequency between unit and audio
stride = 320 # samples
rel_fs = stride / sr

# Number of beams
beam = 1

# Init audio loader
audio_loader = AudioLoader(
    [os.path.join(data_path, "swb1_d{0}//data".format(i)) for i in range(1, 5)],
    sr=sr
)

# Init transcript loader
transcript_loader = TranscriptLoader(os.path.join(data_path, "swb_ms98_transcriptions"))

# Load transcripts
transcriptA = transcript_loader.get_transcript(conversation_id, "A")
transcriptB = transcript_loader.get_transcript(conversation_id, "B")


# Load HuBERT
(
    hubert_model,
    hubert_cfg,
    hubert_task,
) = fairseq.checkpoint_utils.load_model_ensemble_and_task([hubert_path])
hubert_model = hubert_model[0]

# Set Hubert model to eval mode and move to GPU
hubert_model.eval()
hubert_model.cuda()

# Define quantizer
kmeans_quantizer = ApplyKmeans(
    km_path=kmeans_path
)

# Load DLM
speech_dlm = SpeechDLM.from_pretrained(
                model_name_or_path='model_files',
                checkpoint_file='speech_dlm_base.pt',
                data_name_or_path='model_files'
            )

# Set DLM to eval mode and move to GPU
speech_dlm.eval()
speech_dlm.cuda()

# Track gradients of input to decoder
speech_dlm.models[0].decoder.track_input_gradients()

# Define vocoder
decoder = HifiganVocoder(
    vocoder_path = vocoder_path,
    vocoder_cfg_path = vocoder_cfg_path,
)

# Load audio
audio_input = torch.Tensor(
    audio_loader.load_audiofile("sw0{0}.sph".format(conversation_id))
)[:,int(start*sr):int(end*sr)].cuda()


##################### Start of calculation #####################

# Extract features with Hubert
feats, _ = hubert_model.extract_features(
    source=audio_input,
    padding_mask=None,
    mask=False,
    output_layer=hubert_layer # 12
)

feats_A = feats[0] # Features for speaker A
feats_B = feats[1] # Features for speaker B

quantized_units_A, _ = kmeans_quantizer(feats_A) # Quantize features into units
quantized_units_B, _ = kmeans_quantizer(feats_B) # Quantize features into units

quantized_units_A = quantized_units_A.tolist()
quantized_units_B = quantized_units_B.tolist()

str_quantized_units_A = ' '.join(map(str, quantized_units_A)) # Convert units to string
str_quantized_units_B = ' '.join(map(str, quantized_units_B)) # Convert units to string

input_sequences = {
    "unitA": str_quantized_units_A,
    "unitB": str_quantized_units_B
}

# Encode units into DLM units
# (DLM has four extra units <s>, </s>, <pad>, <unk>)
encoded_input_sequences = speech_dlm.encode(input_sequences)

# Generate sequences
generated_sequences = speech_dlm.generate(
    encoded_input_sequences,
    max_len_a = 0,
    max_len_b = max_len,
    sampling=True, # Use nucleus sampling instead of beam search
    beam=beam,
)

# Decode generated sequences into HuBERT units
decoded_generated_sequences = speech_dlm.decode(generated_sequences[0]["tokens"])

# Convert into list of ints
new_units_A = list(map(int, decoded_generated_sequences["unitA"].split(' ')))
new_units_B = list(map(int, decoded_generated_sequences["unitB"].split(' ')))

##################### End of calculation #####################

##################### Start of audio synthesis #####################

###
speaker_ids = [0, 1] # Speaker IDs for resynthesis (not necesarily the same as in the audio)
chunk_length = 501 # Decode audio in chunks to avoid memory error

# Resynthesize units from orignal audio back to audio
resynth_audio_A = np.array([], dtype=np.float32)
resynth_audio_B = np.array([], dtype=np.float32)
for chunk_start in range(0, len(quantized_units_A), chunk_length):
    if chunk_start + 500 > len(quantized_units_A):
        chunk_end = len(quantized_units_A)
    else:
        chunk_end = chunk_start + chunk_length

    codes = [quantized_units_A[chunk_start:chunk_end], quantized_units_B[chunk_start:chunk_end]]
    resynth_audio_chunk_A, resynth_audio_chunk_B = decoder.codes2wav(codes, speaker_ids=speaker_ids)

    resynth_audio_A = np.append(resynth_audio_A, resynth_audio_chunk_A)
    resynth_audio_B = np.append(resynth_audio_B, resynth_audio_chunk_B)
resynth_audio_both = np.asarray([resynth_audio_A, resynth_audio_B]).T

print("resynth_audio_both", resynth_audio_both.shape, resynth_audio_both.dtype)

generated_audio_A = np.array([], dtype=np.float32)
generated_audio_B = np.array([], dtype=np.float32)
for chunk_start in range(0, len(new_units_A), chunk_length):
    if chunk_start + chunk_length > len(new_units_A):
        chunk_end = len(new_units_A)
    else:
        chunk_end = chunk_start + chunk_length

    generated_codes = [new_units_A[chunk_start:chunk_end], new_units_B[chunk_start:chunk_end]]
    generated_audio_chunk_A, generated_audio_chunk_B = decoder.codes2wav(generated_codes, speaker_ids=speaker_ids)

    generated_audio_A = np.append(generated_audio_A, generated_audio_chunk_A)
    generated_audio_B = np.append(generated_audio_B, generated_audio_chunk_B)
generated_audio_both = np.asarray([generated_audio_A, generated_audio_B]).T

print("generated_audio_both", generated_audio_both.shape, generated_audio_both.dtype)

# Get original audio
np_audio_input = audio_input.cpu().numpy()
original_audio_A = np_audio_input[0]
original_audio_B = np_audio_input[1]
original_audio_both = np_audio_input.T

print("original_audio_both", original_audio_both.shape, original_audio_both.dtype)

##################### End of audio synthesis #####################

##################### Start of plotting and saving #####################

# Overwrite previous result audio files in non-run specific folder
wavfile.write("results//audio//resynth_audio_A.wav", sr, resynth_audio_A)
wavfile.write("results//audio//resynth_audio_B.wav", sr, resynth_audio_B)
wavfile.write("results//audio//resynth_audio_both.wav", sr, resynth_audio_both)
wavfile.write("results//audio//generated_audio_A.wav", sr, generated_audio_A)
wavfile.write("results//audio//generated_audio_B.wav", sr, generated_audio_B)
wavfile.write("results//audio//generated_audio_both.wav", sr, generated_audio_both)
wavfile.write("results//audio//original_audio_A.wav", sr, original_audio_A)
wavfile.write("results//audio//original_audio_B.wav", sr, original_audio_B)
wavfile.write("results//audio//original_audio_both.wav", sr, original_audio_both)


# Create folder for run
result_folder = "results"
result_path = os.path.join(result_folder, run_title)
while True:
    try:
        os.mkdir(result_path)
        break
    except FileExistsError:
        ans = input("Run title: \'{0}\' already exists. Overwrite? (y/n)".format(run_title))
        if ans == "y":
            shutil.rmtree(result_path)
            os.mkdir(result_path)
            break
        elif ans == "n":
            run_title = input("New run title: ")
            result_path = os.path.join(result_folder, run_title)
            
# Save results
result_container = SaliencyResult(
    generated_sequences = generated_sequences, # Contians generated sequences, scores, gradients, etc.
    transcripts = [transcriptA, transcriptB],
    start = start,
    end = end,
    src_length = len(encoded_input_sequences["unitA"]),
    tgt_length = max_len,
    rel_fs = rel_fs,
    audio_file = "sw04023.sph"
)

# Save result container as pickle-file
with open(os.path.join(result_path, "result_container.pkl"), "wb") as f:
    pickle.dump(result_container, f)

# Save info as json-file
parameter_dict = {
    "run_title": run_title,
    "sr": sr,
    "conversation_id": conversation_id,
    "audio_duration": audio_duration,
    "start": start,
    "end": end,
    "max_len": max_len,
    "src_length": len(encoded_input_sequences["unitA"]),
    "rel_fs": rel_fs,
    "beam": beam
}
with open(os.path.join(result_path, "info.json"), "w") as f:
    json.dump(parameter_dict, f, indent=4)

# Save audio to run folder
wavfile.write(os.path.join(result_path, "resynth_audio_both.wav"), sr, resynth_audio_both)
wavfile.write(os.path.join(result_path, "generated_audio_both.wav"), sr, generated_audio_both)
wavfile.write(os.path.join(result_path, "generated_audio_A.wav"), sr, generated_audio_A)
wavfile.write(os.path.join(result_path, "generated_audio_B.wav"), sr, generated_audio_B)
wavfile.write(os.path.join(result_path, "original_audio_both.wav"), sr, original_audio_both)

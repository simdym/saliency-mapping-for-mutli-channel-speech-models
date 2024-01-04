import torch
import math

from transcript_utils.transcript import Transcript

from typing import List

def assign_word_saliency(
        saliency: torch.Tensor, # (generated_outputs, channels, original_input)
        transcripts: List[Transcript],
        start: float,
        end: float,
        transcript_sr: int=16000,
        rel_fs: float=1.0 # saliency frequency / auido sample rate
    ):
    """
    Assigns saliency to the words in the given transcripts.
    """
    if saliency.size(1) != len(transcripts):
        raise ValueError("Saliency values must have the same number of channels as there are transcripts")

    saliency_result = {
        channel: {
            "words": [],
            "saliencies": []
        } for channel in range(len(transcripts))
    }

    for channel in range(len(transcripts)):
        for word in transcripts[channel].get_elements_between(start, end):
            word_saliencies = []
            for generated_output in range(saliency.size(0)):
                word_start = (word.start-start)/rel_fs
                word_end = (word.end-start)/rel_fs

                #Extract fully contained unit saliencies
                word_saliency = saliency[generated_output, channel, math.ceil(word_start):math.floor(word_end)+1].sum()

                #Extract partially contained unit saliencies
                front_unit_ratio = (word_start - math.ceil(word_start)) * rel_fs
                back_unit_ratio = (math.floor(word_end) - word_end) * rel_fs

                word_saliency += saliency[generated_output, channel, math.ceil(word_start)-1] * front_unit_ratio
                word_saliency += saliency[generated_output, channel, math.floor(word_end)] * back_unit_ratio

                word_saliencies.append(word_saliency)
            
            word_saliencies = torch.Tensor(word_saliencies)

            saliency_result[channel]["words"].append(word.content)
            saliency_result[channel]["saliencies"].append(word_saliencies) # word_saliencies dim: (generated_outputs)

        saliency_result[channel]["saliencies"] = torch.stack(saliency_result[channel]["saliencies"], dim=0).softmax(dim=1) # dim: (words, generated_outputs)
    
    return saliency_result
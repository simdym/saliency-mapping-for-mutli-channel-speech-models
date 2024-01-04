import torch

from transcript_utils.transcript import Transcript

from typing import List

class SaliencyResult():
    def __init__(
            self,
            generated_sequences: torch.Tensor,
            transcripts: List[Transcript],
            start: int,
            end: int,
            src_length: int,
            tgt_length: int,
            rel_fs: float,
            audio_file: str
        ) -> None:
        self.generated_sequences = generated_sequences
        self.transcripts = transcripts
        self.start = start
        self.end = end
        self.src_length = src_length
        self.tgt_length = tgt_length
        self.rel_fs = rel_fs
        self.audio_file = audio_file
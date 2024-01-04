from transcript_utils.transcript import Transcript
from transcript_utils.file_utils import load_isip_transcript
import os

class TranscriptLoader():
    def __init__(self, transcript_folder: str) -> None:
        self.transcript_folder = transcript_folder

    
    def get_transcript(self, id: str, channel: str, mode: str="word") -> Transcript:
        """
        Get transcript from file with given id, channel and mode

        Args:
            id: id of transcript
            channel: either 'A' or 'B'
            mode: either 'word' or 'trans"
        """

        #"swb_ms98_transcriptions//40//4023//sw4023A-ms98-a-word.text"
        transcript_path = os.path.join(self.transcript_folder, id[:2], id, f"sw{id}{channel}-ms98-a-{mode}.text")
        return load_isip_transcript(transcript_path)


if __name__ == "__main__":
    data_path = "//talebase//data//speech_raw//switchboard_1"
    transcript_folder = os.path.join(data_path, "swb_ms98_transcriptions")
    transcript_loader = TranscriptLoader(transcript_folder)
    
    print(transcript_loader.get_transcript("4023", "A", "trans"))
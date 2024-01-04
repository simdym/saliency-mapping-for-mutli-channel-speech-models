from .transcript import Transcript

def load_isip_transcript(file_path: str) -> Transcript:
    """
    Returns the transcript of a given file path saved in the ISIP format.
    """
    transcript = Transcript()

    with open(file_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            line_split = line.split(" ")
            sentence_id = int(line_split[0].split("-")[-1])

            start = float(line_split[1])
            end = float(line_split[2])
            content = " ".join(line_split[3:])

            transcript.add_element(start, end, content, sentence_id)

        return transcript


def load_ws_transcript(file_path: str) -> Transcript:
    """
    Returns the transcript of a given file path in the WaveSurfer format.

    WS format does not include sentence IDs
    """
    transcript = Transcript()

    with open(file_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            line_split = line.split(" ")

            start = float(line_split[0])
            end = float(line_split[1])
            content = " ".join(line_split[2:])

            transcript.add_element(start, end, content)

        return transcript


def save_ws_transcript(transcript: Transcript, file_path: str):
    """
    Saves a transcript in the WaveSurfer format.
    """
    with open(file_path, "w") as f:
        for element in transcript.elements:
            f.write(str(element))

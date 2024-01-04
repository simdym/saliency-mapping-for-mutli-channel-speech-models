import soundfile as sf
import librosa
import numpy as np
import os

from typing import List
from numpy.typing import NDArray

def load_audio(file_path: str, sr: int = None):
    """Loads audio file and resamples it"""
    raw_audio, raw_sr = sf.read(file_path)

    if sr == None:
        resampled_audio = raw_audio
    elif raw_sr != sr:
        resampled_audio = np.asarray([
            librosa.resample(raw_audio[:, 0], orig_sr=raw_sr, target_sr=sr),
            librosa.resample(raw_audio[:, 1], orig_sr=raw_sr, target_sr=sr)
        ])
    else:
        resampled_audio = raw_audio
    
    return resampled_audio


class AudioFolder():
    """Folder with audio-files"""
    def __init__(self, path: str) -> None:
        self.path = path
        self.files = os.listdir(path)
    
    def __len__(self):
        return len(self.files)
    
    def has_file(self, filename: str):
        return filename in self.files

    def get_file(self, filename: str, sr: int = None) -> NDArray:
        return load_audio(os.path.join(self.path, filename), sr)


class AudioLoader():
    """Class for loading audio which may be in different directories"""
    def __init__(self, data_paths: List[str], sr: int = None) -> None:
        """
        Args:
            data_paths: list of paths to folders containing audio-files
            sr: sample rate to resample audio to
        """
        self.audio_folders = [AudioFolder(path) for path in data_paths]
        self.sr = sr
    
    def __len__(self):
        return sum([len(audio_folder) for audio_folder in self.audio_folders])
    
    def get_filenames(self):
        files = []
        for audio_folder in self.audio_folders:
            files.extend(audio_folder.files)
        return files

    def load_audiofile(self, filename: str):
        for audio_folder in self.audio_folders:
            if audio_folder.has_file(filename):
                return audio_folder.get_file(filename, self.sr)
        raise ValueError("No file with name {0} found".format(filename))


if __name__ == "__main__":
    data_path = "//talebase//data//speech_raw//switchboard_1"

    import os

    audio_loader = AudioLoader(
        [os.path.join(data_path, "swb1_d{0}//data".format(i)) for i in range(1, 5)],
        sr=16000
    )


    print(len(audio_loader))
    print(audio_loader.get_filenames())
    print(audio_loader.load_audiofile("sw02001.sph").shape)
    try:
        print(audio_loader.load_audiofile("sw020011.sph").shape)
    except ValueError as e:
        print(e)
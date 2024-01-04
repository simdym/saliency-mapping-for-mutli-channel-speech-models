from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, audio_loader) -> None:
        self.audio_loader = audio_loader
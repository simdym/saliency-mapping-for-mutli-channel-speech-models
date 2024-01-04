from typing import List

class TranscriptElement():
    def __init__(self, start: float, end: float, content: str, sentence_id: int = None):
        if end < start:
            raise ValueError("End time is before start time")
        
        self.start = start
        self.end = end
        self.content = content
        self.sentence_id = sentence_id
    
    def __str__(self):
        return f"{self.start} {self.end} {self.content}"

class Transcript:
    def __init__(self):
        self.elements = []

    def __str__(self):
        string = ""
        for e in self.elements:
            string += str(e)
        return string
    
    def __len__(self):
        return len(self.elements)

    def add_element(self, start, end, content, sentence_id=None) -> None:
        self.elements.append(TranscriptElement(start, end, content, sentence_id))
    
    def get_elements_between(self, start, end, include_edge=False) -> List[TranscriptElement]:
        if include_edge:
            return [e for e in self.elements if e.start < end and e.end > start]
        else:
            return [e for e in self.elements if e.start >= start and e.end <= end]
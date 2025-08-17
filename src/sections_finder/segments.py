from typing import Optional, List


class Segment:
    def __init__(self, type: str, start: Optional[int] = None, end: Optional[int] = None):
        self.type = type
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Segment(type={self.type}, start={self.start}, end={self.end})"
    


class Segments:
    def __init__(self, segments: Optional[List[Segment]] = None):
        self.segments = segments if segments is not None else []
    
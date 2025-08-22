class DetectedObject:
    """
    Represents a detected object for tracking purposes.

    This class encapsulates the state of a detected object including its unique
    identifier, bounding box coordinates, and tracking statistics to maintain
    continuity across video frames.
    """

    def __init__(
        self, idx: int, box: list, age: int = 1, unmatched_age: int = 0
    ) -> None:
        """
        Initialize a detected object with tracking information.

        Args:
            idx (int): Unique identifier for the detected object.
            box (list): Bounding box coordinates in format [x1, y1, x2, y2].
            age (int): Number of consecutive frames this object has been matched (default: 1).
            unmatched_age (int): Number of consecutive frames this object has been unmatched (default: 0).
        """
        self.idx = idx  # Unique object identifier
        self.box = box  # Bounding box coordinates [x1, y1, x2, y2]
        self.age = age  # Consecutive matched frames (indicates tracking stability)
        self.unmatched_age = unmatched_age  # Consecutive unmatched frames

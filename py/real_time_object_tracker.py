import copy
import math

import cv2 as cv
import numpy as np
from detected_object import DetectedObject
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from utils import division_by_zero, id_to_color


class RealTimeObjectTracker:
    # Tracking configuration constants
    MIN_HIT_STREAK = 1  # Minimum consecutive matches before displaying tracked object
    MAX_UNMATCHED_AGE = 2  # Maximum consecutive unmatched frames before removing track

    def __init__(
        self,
        model: YOLO,
        image_width: int,
        image_height: int,
        conf: float = 0.5,
        iou: float = 0.4,
    ) -> None:
        """
        Initialize the real-time object tracking system.

        Args:
            model (YOLO): Pre-trained YOLO model for object detection.
            image_width (int): Width of the input images in pixels.
            image_height (int): Height of the input images in pixels.
            conf (float): Confidence threshold for object detection (default: 0.5).
            iou (float): IoU threshold for non-maximum suppression (default: 0.4).
        """
        self.idx: int = 0  # Counter for generating unique track IDs
        # List of currently tracked objects
        self.detected_objects: list[DetectedObject] = []

        # YOLO model configuration
        self.model = model
        self.image_width = image_width  # Used for normalization in cost metrics
        self.image_height = image_height  # Used for normalization in cost metrics
        self.conf = conf  # Detection confidence threshold
        self.iou = iou  # Non-maximum suppression IoU threshold

    def inference(self, input_image) -> tuple[cv.Mat, list[DetectedObject]]:
        """
        Perform complete object tracking inference on an input image.

        This method runs object detection, associates detections with existing tracks,
        updates tracking states, and draws bounding boxes with track IDs on the image.

        Args:
            input_image: Annotated image for object detection and tracking.
            show (bool): Whether to draw bounding boxes and track IDs on the image (default: False).

        Returns:
            tuple: A tuple containing (original_image, detected_objects)
                - annotated_image (cv.Mat):
                - detected_objects (list[DetectedObject]): List of currently tracked objects
        """
        # Create a copy to avoid modifying the original image
        image = copy.deepcopy(input_image)
        # Run object detection on the image
        out_boxes, _, _ = self.predict(image)

        # Initialize list for updated tracked objects
        new_detected_objects: list[DetectedObject] = []

        # Extract bounding boxes from currently tracked objects
        old_detected_objects_boxes = [obj.box for obj in self.detected_objects]

        # Perform data association between old tracks and new detections
        matches, unmatched_detections, unmatched_trackers = self.associate(
            old_detected_objects_boxes, out_boxes
        )

        # Process matched detections - update existing tracks
        for match in matches:
            oldIndex, newIndex = match[0], match[1]
            detected_obj = DetectedObject(
                self.detected_objects[oldIndex].idx,  # Keep existing track ID
                out_boxes[newIndex],  # Update with new bounding box
                self.detected_objects[oldIndex].age + 1,  # Increment age
            )
            new_detected_objects.append(detected_obj)

        # Process new (unmatched) detections - create new tracks
        for new_box in unmatched_detections:
            self.idx += 1  # Generate new unique track ID
            detected_obj = DetectedObject(self.idx, new_box)
            new_detected_objects.append(detected_obj)

        # Process unmatched trackers - increment unmatched age
        for old_box in unmatched_trackers:
            try:
                index = old_detected_objects_boxes.index(old_box)
                obj = self.detected_objects[index]
                obj.unmatched_age += 1  # Increment frames without match
                new_detected_objects.append(obj)
            except ValueError:
                # Box not found in old_detected_objects_boxes - skip it
                continue

        # Remove objects that exceeded unmatched age threshold
        for obj in new_detected_objects:
            if obj.unmatched_age > self.MAX_UNMATCHED_AGE:
                new_detected_objects.remove(obj)

        # Update internal tracking state
        self.detected_objects = new_detected_objects

        for obj in self.detected_objects:
            # Draw bounding box and ID for stable tracks
            if obj.age >= self.MIN_HIT_STREAK:
                left, top, right, bottom = obj.box

                left, top = int(left), int(top)
                right, bottom = int(right), int(bottom)

                # Draw bounding box
                cv.rectangle(
                    image,
                    (left, top),
                    (right, bottom),
                    id_to_color(obj.idx * 10),  # Color based on track ID
                    thickness=3,
                )
                # Draw track ID label
                cv.putText(
                    image,
                    str(obj.idx),
                    (left - 10, top - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    id_to_color(obj.idx * 10),
                    thickness=2,
                )

        return image, new_detected_objects

    def associate(self, old_boxes, new_boxes, iou_thresh=0.3):
        """
        Associate old bounding boxes with new bounding boxes using Hungarian algorithm.

        This method performs data association between detections from consecutive frames
        to maintain object tracking continuity. It uses a cost matrix based on IoU and
        other metrics, then applies the Hungarian algorithm for optimal assignment.

        Args:
            old_boxes (list): Former bounding boxes from previous frame [x1, y1, x2, y2].
            new_boxes (list): New bounding boxes from current frame [x1, y1, x2, y2].
            iou_thresh (float): Minimum IoU threshold for valid matches (default: 0.3).

        Returns:
            tuple: (matches, unmatched_detections, unmatched_trackers)
                - matches: Array of matched indices [[old_idx, new_idx], ...]
                - unmatched_detections: List of new boxes without matches
                - unmatched_trackers: List of old boxes without matches
        """
        old_box_length = len(old_boxes)
        new_box_length = len(new_boxes)

        # Handle edge cases with empty box lists
        if old_box_length == 0 and new_box_length == 0:
            return [], [], []
        elif len(old_boxes) == 0:
            return [], new_boxes, []
        elif len(new_boxes) == 0:
            return [], [], old_boxes

        # Create cost matrix (old_boxes x new_boxes) for Hungarian algorithm
        iou_matrix = np.zeros((old_box_length, new_box_length), dtype=np.float32)

        # Populate cost matrix with similarity scores between all box pairs
        for i, old_box in enumerate(old_boxes):
            for j, new_box in enumerate(new_boxes):
                iou_matrix[i][j] = self.total_cost(old_box, new_box)

        # Apply Hungarian algorithm to find optimal assignment (minimize negative cost)
        hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
        hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

        # Initialize result lists
        matches, unmatched_detections, unmatched_trackers = [], [], []

        # Process Hungarian algorithm results and filter by IoU threshold
        for row in hungarian_matrix:
            x, y = row[0], row[1]
            if iou_matrix[x, y] < iou_thresh:
                # Assignment below threshold - treat as unmatched
                unmatched_trackers.append(old_boxes[x])
                unmatched_detections.append(new_boxes[y])
            else:
                # Valid assignment - add to matches
                matches.append(row.reshape(1, 2))

        # Convert matches to proper numpy array format
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.vstack(matches)

        # Find old boxes that weren't assigned to any new box
        for index, box in enumerate(old_boxes):
            if index not in hungarian_matrix[:, 0]:
                unmatched_trackers.append(box)

        # Find new boxes that weren't assigned to any old box
        for index, box in enumerate(new_boxes):
            if index not in hungarian_matrix[:, 1]:
                unmatched_detections.append(box)

        return matches, unmatched_detections, unmatched_trackers

    def predict(self, input_image: cv.Mat | np.ndarray):
        """
        Run object detection inference on the input image.

        Args:
            input_image (cv.Mat | np.ndarray): Input image for object detection.

        Returns:
            tuple: A tuple containing (boxes, scores, categories) if objects are detected,
                   otherwise returns ([], [], []).
        """
        # Run YOLO model prediction on the input image
        results = self.model.predict(input_image, conf=self.conf)

        # Extract the first (and typically only) result
        result = results[0]
        if result.boxes:
            # Convert bounding boxes to integer coordinates [x1, y1, x2, y2]
            box = [[float(j) for j in i] for i in result.boxes.xyxy]
            # Extract confidence scores for each detection
            score = result.boxes.conf
            # Extract class categories for each detection
            category = result.boxes.cls

            return box, score, category

        # Return empty lists if no objects detected
        return [], [], []

    def box_iou(self, prev_box, current_box) -> float:
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        Args:
            prev_box (list or tuple): [x1, y1, x2, y2] coordinates of the previous bounding box.
            current_box (list or tuple): [x1, y1, x2, y2] coordinates of the current bounding box.

        Returns:
            float: IoU value between the two boxes (0.0 to 1.0).
        """
        # Determine the coordinates of the intersection rectangle
        xA = max(prev_box[0], current_box[0])  # Intersection top-left x
        yA = max(prev_box[1], current_box[1])  # Intersection top-left y
        xB = min(prev_box[2], current_box[2])  # Intersection bottom-right x
        yB = min(prev_box[3], current_box[3])  # Intersection bottom-right y

        # Compute the area of intersection rectangle (W * H)
        intercept_area = max(0.0, xB - xA) * max(0.0, yB - yA)

        # Compute the area of both bounding boxes (W * H)
        prev_box_area = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
        current_box_area = (current_box[2] - current_box[0]) * (
            current_box[3] - current_box[1]
        )

        # Compute the union area
        union_area = (prev_box_area + current_box_area) - intercept_area

        # Compute the IoU
        iou = intercept_area / float(division_by_zero(union_area))

        return iou

    def convert_box_from_xyxy_to_xywh(
        self, bounding_box
    ) -> tuple[float, float, float, float]:
        """
        Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

        Returns:
            Boxes in [x_center, y_center, width, height] format, where x_center, y_center are
            the coordinates of the center point of the bounding box, width, height are the
            dimensions of the bounding box.
        """

        # Calculate width and height of the bounding box
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]
        # Calculate center coordinates
        x_center = bounding_box[0] + (width / 2.0)
        y_center = bounding_box[1] + (height / 2.0)

        return (x_center, y_center, width, height)

    def sanchez_matilla(self, prev_box, current_box) -> float:
        """
        Calculate the linear cost between two bounding boxes using the Sanchez-Matilla metric.

        Args:
            prev_box (list or tuple): [x1, y1, x2, y2] coordinates of the previous bounding box.
            current_box (list or tuple): [x1, y1, x2, y2] coordinates of the current bounding box.

        Returns:
            float: Linear cost value between the two boxes.
        """
        # Sanchez-Matilla metric formula:
        # C(A, B) = cdist(A, B) * cshp(A, B)
        # where:
        # cdist(A, B) = Qdist / sqrt((XA - XB)^2 + (YA - YB)^2)           Distance Measure
        # cshp(A, B) =  Qshp / sqrt((HA - HB)^2 + (WA - WB)^2)            Shape Measure

        # Q_dist: Diagonal length of the image (Pythagoras Theorem)
        Q_dist = math.sqrt(self.image_width**2 + self.image_height**2)
        # Q_shape: Area of the image
        Q_shape = self.image_width * self.image_height

        # Convert bounding boxes to center coordinates and dimensions
        xA, yA, wA, hA = self.convert_box_from_xyxy_to_xywh(prev_box)
        xB, yB, wB, hB = self.convert_box_from_xyxy_to_xywh(current_box)

        # C_dist: Normalized inverse distance between box centers
        C_dist = Q_dist / division_by_zero(math.sqrt((xA - xB) ** 2 + (yA - yB) ** 2))
        # C_shape: Normalized inverse difference in box shapes
        C_shape = Q_shape / division_by_zero(math.sqrt((hA - hB) ** 2 + (wA - wB) ** 2))

        # Final linear cost is the product of the two metrics
        linear_cost = C_dist * C_shape

        return linear_cost

    def yu(self, prev_box, current_box, w1=0.5, w2=1.5) -> float:
        """
        Calculate the exponential cost between two bounding boxes using the Yu metric.

        Args:
            prev_box (list or tuple): [x1, y1, x2, y2] coordinates of the previous bounding box.
            current_box (list or tuple): [x1, y1, x2, y2] coordinates of the current bounding box.
            w1 (float): Weight for distance component (default: 0.5).
            w2 (float): Weight for shape component (default: 1.5).

        Returns:
            float: Exponential cost value between the two boxes.
        """
        # Yu metric formula:
        # C(A, B) = capp(A, B) * cdist(A, B) * cshp(A, B)
        # where:
        # capp(A, B) = cos(featA, featB)                                   Appearance Measure
        # cdist(A, B) = e^(-w1*((XA-XB/WA)^2 + (YA-YB/HA)^2))              Distance Measure
        # cshp(A, B) = e^(-w2*(|HA-HB|/(HA+HB) + |WA-WB|/(WA+WB)))         Shape Measure

        # Convert bounding boxes to center coordinates and dimensions
        xA, yA, wA, hA = self.convert_box_from_xyxy_to_xywh(prev_box)
        xB, yB, wB, hB = self.convert_box_from_xyxy_to_xywh(current_box)

        # Normalized squared distance in x direction
        A1 = ((xA - xB) / division_by_zero(wA)) ** 2
        # Normalized squared distance in y direction
        A2 = ((yA - yB) / division_by_zero(hA)) ** 2

        # Combined normalized distance metric
        A = A1 + A2
        # Shape difference metric (relative height and width changes)
        B = (abs(hA - hB) / (hA + hB)) + (abs(wA - wB) / (wA + wB))

        # Exponential decay for distance similarity
        C_dist = math.exp(-w1 * A)
        # Exponential decay for shape similarity
        C_shape = math.exp(-w2 * B)

        # Final exponential cost is the product of the two metrics
        # This implementation will ignore the appearance measure
        exponential_cost = C_dist * C_shape

        return exponential_cost

    def total_cost(
        self,
        old_box,
        new_box,
        iou_thresh=0.3,
        linear_thresh=10000,
        exp_thresh=0.5,
    ) -> float:
        """
        Calculate the combined cost for tracking between two bounding boxes.

        This method combines IoU, linear (Sanchez-Matilla), and exponential (Yu) metrics
        to determine if two boxes should be matched for tracking. Returns IoU score if
        all thresholds are met, otherwise returns 0.

        Args:
            old_box (list or tuple): [x1, y1, x2, y2] coordinates of the previous bounding box.
            new_box (list or tuple): [x1, y1, x2, y2] coordinates of the current bounding box.
            image_width (int): Width of the image.
            image_height (int): Height of the image.
            iou_thresh (float): Minimum IoU threshold for matching (default: 0.3).
            linear_thresh (float): Minimum linear cost threshold (default: 10000).
            exp_thresh (float): Minimum exponential cost threshold (default: 0.5).

        Returns:
            float: IoU score if all thresholds are met, otherwise 0.0.
        """
        # Calculate IoU similarity metric
        iou_score = self.box_iou(old_box, new_box)
        # Calculate Sanchez-Matilla linear cost metric
        linear_cost = self.sanchez_matilla(old_box, new_box)
        # Calculate Yu exponential cost metric
        exponential_cost = self.yu(old_box, new_box)

        # Check if all metrics meet their respective thresholds
        if (
            iou_score >= iou_thresh
            and linear_cost >= linear_thresh
            and exponential_cost >= exp_thresh
        ):
            return iou_score  # Return IoU as the final matching score
        else:
            return 0.0  # No match if any threshold is not met

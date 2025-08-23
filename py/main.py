import glob
import os

import cv2 as cv
from real_time_object_tracker import RealTimeObjectTracker
from ultralytics import YOLO


def YOLO11(images_path):
    model = YOLO("yolo11s.pt")

    for image_file in images_path:
        results = model.track(image_file, conf=0.5, persist=True)

        if len(results) > 0:
            result = results[0]

            if result.boxes and result.boxes.is_track:

                # Visualize the result on the frame
                annotated_frame = result.plot(conf=False)

            # Display the annotated frame
            cv.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    # Clean up OpenCV windows
    cv.destroyAllWindows()


def simpleSortAlgorithm(images_path):
    model = YOLO("yolo11s.pt")

    height, width = 360, 1280
    tracker = RealTimeObjectTracker(model, width, height)

    for image_file in images_path:
        image = cv.imread(image_file)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        output_image, _ = tracker.inference(image)

        new_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
        cv.imshow("Object Tracking", new_image)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up OpenCV windows
    cv.destroyAllWindows()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    images_path = sorted(glob.glob(os.path.join(data_dir, "*.png")))

    YOLO11(images_path)

    # simpleSortAlgorithm(images_path)

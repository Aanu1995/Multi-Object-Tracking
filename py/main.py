import glob
import os

import cv2 as cv
from real_time_object_tracker import RealTimeObjectTracker
from ultralytics import YOLO


def main():
    model = YOLO("yolo11s.pt")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    images_path = sorted(glob.glob(os.path.join(data_dir, "*.png")))

    height, width = 360, 1280
    tracker = RealTimeObjectTracker(model, width, height)

    for image_file in images_path:
        image = cv.imread(image_file)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        output_image, _ = tracker.inference(image)

        # show tracking at 15 FPS
        new_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
        cv.imshow("Object Tracking", new_image)

        # Wait for 1/30 second (66.67 ms) to achieve 30 FPS
        key = cv.waitKey(33)
        if key == ord("q") or key == 27:  # 'q' key or ESC to quit
            break

    # Clean up OpenCV windows
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

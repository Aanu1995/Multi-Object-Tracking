import random

import numpy as np
from scipy.optimize import linear_sum_assignment

# Detections at time 0
A = [100, 120, 130, 330]
B = [300, 350, 400, 400]
C = [577, 138, 709, 244]

# Detections at time 1
D = [50, 400, 100, 550]  # Should no frame
E = [99, 120, 132, 333]  # Should match frame A
F = [302, 352, 406, 400]  # Shold match frame B

old = [A, B, C]
new = [D, E, F]


### HELPER IOU FUNCTION
def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # abs((xi2 - xi1)*(yi2 - yi1))
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)

    # abs((box1[3] - box1[1])*(box1[2]- box1[0]))
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    # abs((box2[3] - box2[1])*(box2[2]- box2[0]))
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = (box1_area + box2_area) - inter_area

    # compute the IoU
    iou = inter_area / float(union_area)
    return iou


iou_matrix = np.zeros((len(old), len(new)), dtype=np.float32)

# Go through old boxes and new boxes and compute an IOU to store in a matrix
for i, old_box in enumerate(old):
    for j, new_box in enumerate(new):
        iou_matrix[i][j] = box_iou(old_box, new_box)

# Perform Non-Maximum Suppression (NMS)
# Go through the IOU matrix and replace positive values with 1
# Always take the maximum value (if there are two positive values)
for idx, iou in enumerate(iou_matrix):
    iou_matrix[idx] = [1 if (x == max(iou) and max(iou) > 0) else 0 for x in iou]

# Call the Linear Assignment Method (Hungarian Algorithm)
# Watch for the minimization vs maximization assignment problem
hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)

hungarian = np.array(list(zip(hungarian_row, hungarian_col)))

# Declare a list for matches, unmatched detections, and unmatched trackings
matches = []
unmatched_trackers, unmatched_detections = [], []

# Go through the hungarian matrix
# Take the match using the old and new boxes and the match indications
# Check that the IOU is > 0 and keep the bounding box if so
# Otherwise add it to unmatched detections and trackings

default_iou = 0.3  # boxes overlap if iOU > 0.3
for h in hungarian:
    if iou_matrix[h[0], h[1]] < default_iou:
        unmatched_trackers.append(old[h[0]])
        unmatched_detections.append(new[h[1]])
    else:
        matches.append(h.reshape(1, 2))

if len(matches) == 0:
    matches = np.empty((0, 2), dtype=int)
else:
    matches = np.vstack(matches)


# Go through old bounding boxes and add old boxes that didn't match to unmatched trackers
for t, trk in enumerate(old):
    if t not in hungarian[:, 0]:
        unmatched_trackers.append(trk)

# Do the same for new detections
for d, det in enumerate(old):
    if d not in hungarian[:, 1]:
        unmatched_trackers.append(det)

# Now, we want to display the matched bounding boxes
# Display everything properly
display_match = []
for match in matches:
    display_match.append((new[match[1]], old[match[0]]))

print("Matched Detections")
print(display_match)
print("Unmatched Detections ")
print(np.array(unmatched_detections))
print("Unmatched trackers ")
print(np.array(unmatched_trackers))

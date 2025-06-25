import cv2
import numpy as np
from sort import Sort
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
model.fuse()  # Speed up inference
model.overrides['verbose'] = False  # Disable logs

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Video source
cap = cv2.VideoCapture("D:/OpenCV/Vehicle Detection/traffic.mp4")
if not cap.isOpened():
    print("Error opening video")
    exit()

# Counting line setup
count_line_position = 550
offset = 6
counted_ids = set()

# Class names (YOLO class index to label)
class_names = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
    21: "ambulance"  # custom trained ID
}

# ID to class map
id_class_map = {}
vehicle_counts = {name: 0 for name in class_names.values()}

# Start reading video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    detections = []
    classes = []

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        cls_id = int(box.cls[0])
        if cls_id in class_names:
            detections.append([x1, y1, x2, y2, score])
            classes.append(cls_id)

    detections_np = np.array(detections) if detections else np.empty((0, 5))
    tracked_objects = tracker.update(detections_np)

    for i, obj in enumerate(tracked_objects):
        x1, y1, x2, y2, track_id = obj.astype(int)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Match track_id to its class
        if i < len(classes):  # Safe check
            class_id = classes[i]
            id_class_map[track_id] = class_id
        else:
            class_id = id_class_map.get(track_id, -1)

        # Draw bounding box and label
        color = (0, 255, 0)
        label = class_names.get(class_id, "vehicle")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Count if it crosses the line and not counted before
        if track_id not in counted_ids:
            if count_line_position - offset < cy < count_line_position + offset:
                counted_ids.add(track_id)
                vehicle_counts[label] += 1

    # Draw counting line
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 0, 0), 2)

    # Display total count
    y_offset = 30
    for i, (veh_type, count) in enumerate(vehicle_counts.items()):
        cv2.putText(frame, f"{veh_type}: {count}", (50, y_offset + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Vehicle Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

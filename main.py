import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import json

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('RGB', RGB)
cv2.moveWindow('RGB', 100, 100) 

cap = cv2.VideoCapture('parking1.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Read vacant spaces from JSON file
with open("sample.json", "r") as json_file:
    vacant_spaces_data = json.load(json_file)

# Convert the vacant_spaces_data dictionary into a list of areas
areas = []
for area_index, vacant_space_coordinates in vacant_spaces_data.items():
    areas.append(np.array(vacant_space_coordinates, np.int32))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)

    for area_index, area in enumerate(areas):
        area_list = []
       # results = model.predict(frame)
        a = results[0].boxes.data.numpy()  # Extract NumPy array from Boxes object
        px = pd.DataFrame(a, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class_id']).astype("float")

        for index, row in px.iterrows():
            x1 = int(row.iloc[0])
            y1 = int(row.iloc[1])
            x2 = int(row.iloc[2])
            y2 = int(row.iloc[3])
            d = int(row.iloc[5])

            c = class_list[d]
            if 'car' in c:
                cx = int(x1 + x2) // 2
                cy = int(y1 + y2) // 2

                results_area = cv2.pointPolygonTest(area, ((cx, cy)), False)
                if results_area >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    area_list.append(c)

        area_count = len(area_list)

        if area_count == 1:
            cv2.polylines(frame, [area], True, (0, 0, 255), 2)
            cv2.putText(frame, str(area_index + 1), (area[0][0], area[0][1] - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.polylines(frame, [area], True, (0, 255, 0), 2)
            cv2.putText(frame, str(area_index + 1), (area[0][0], area[0][1] - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

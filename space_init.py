import cv2
import json
import numpy as np

# Global variables
vacant_spaces = {}
current_area = 1
current_points = []
current_pointss = []

def mouse_callback(event, x, y, flags, param):
    global current_area, current_points, current_pointss
    

    # **Scale the mouse coordinates to the original frame size:**
    orig_x = int(x * 1020/frame.shape[1] )  # Scale x to original width
    orig_y = int(y * 500/frame.shape[0] )   # Scale y to original height

    if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 4:
        current_points.append([orig_x, orig_y])
        current_pointss.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        if len(current_points) == 4:
            # Save the polygon points
            vacant_spaces[f"area{current_area}"] = current_points
            current_area += 1

            # Draw the green polygon
            pts = np.array(current_pointss, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            current_points = []
            current_pointss = []

            cv2.imshow("Video", frame)

# Read video file
video_path = 'parking1.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Capture the first frame
ret, frame = cap.read()

# Check if the frame is captured successfully
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Resize the window
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 1020, 500)
cv2.moveWindow("Video", 100, 100)
# Create a window and set the callback function
cv2.setMouseCallback("Video", mouse_callback)

# Display the first frame
cv2.imshow("Video", frame)

# Wait for the user to press 's' to save and exit
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == 27:  # 's' key or ESC key
        break

# Release video capture object and destroy windows
cap.release()
cv2.destroyAllWindows()

# Save the vacant_spaces dictionary to a JSON file
with open('sample.json', 'w') as json_file:
    json.dump(vacant_spaces, json_file, indent=4)

print("Polygon points saved to sample.json.")

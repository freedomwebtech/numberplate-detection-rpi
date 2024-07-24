import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import os
import pytesseract
from datetime import datetime

model = YOLO('best.pt')

cap = cv2.VideoCapture('mycarplate.mp4')

# Load class names from coco1.txt
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Create output folder with current date
current_date = time.strftime("%Y-%m-%d")
output_folder = f"{current_date}"
os.makedirs(output_folder, exist_ok=True)

# Initialize list to store detected license plate numbers
count = 0

# Function to save data to text file
def save_to_text(number_plate, timestamp):
    text_filename = os.path.join(output_folder, 'numberplates.txt')
    with open(text_filename, 'a') as f:
        f.write(f"Number Plate: {number_plate}, Detected at: {timestamp}\n")
    print(f"Data saved to {text_filename}")

# Define detection area
area = [(0, 364), (1, 416), (1019, 419), (1018, 356)]
def save_to_text(number_plate, timestamp):
    text_filename = os.path.join(output_folder, 'numberplates.txt')
    with open(text_filename, 'a') as f:
        f.write(f"{timestamp}:--{number_plate}\n")
    print(f"Data saved to {text_filename}")
while True:
    ret, frame = cap.read()
    
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))   
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((x1, y1)), False)
        
        if result >= 0:
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (200, 40))
        
            text = pytesseract.image_to_string(crop, config='-l eng --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
            
            # Save to text file
            if text:
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_to_text(text, current_datetime)
                
                # Print and display
                print(f"Detected Number Plate: {text} at {current_datetime}")

            cv2.imshow('crop', crop)
    
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2) 
    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

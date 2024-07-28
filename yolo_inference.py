from ultralytics import YOLO

model = YOLO(r"C:\Users\44780\Documents\new_motion_tracking\models\best.pt")

results = model.predict(r"C:\Users\44780\Documents\new_motion_tracking\input_videos\counter_attack.mp4", save=True)
print(results[0])
print("==================")
for box in results.boxes[0].boxes:
    print(box)
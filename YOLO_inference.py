from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # Load model

result = model.predict('Assets\input_video.mp4', save= True)  # Inference
print(result)
print("Boxes:")
for box in result[0].boxes:
    print(box)
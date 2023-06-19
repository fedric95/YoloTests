from ultralytics import YOLO

#os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMzEyZDhiMC0xZWIxLTRjYzctODcxMi04ZmUyYzc1MmQ1ZDQifQ=="
#os.environ["NEPTUNE_PROJECT"]   = "federico.ricciuti/YOLOv8"

model = YOLO('./model.yaml')     # Build from YAML 
model = model.load('yolov8n.pt') # Transfer weights

# Train the model
model.train(
    data='./dataset.yaml',
    epochs=20,
    imgsz=512,
    workers=0,
    batch = 16
)

import pdb
pdb.set_trace()

# Evaluate the model's performance on the validation set
results = model.val()

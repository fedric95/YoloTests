import matplotlib.pyplot as plt
import pdb

from ultralytics import YOLO
import ultralytics.yolo.engine.model 
from trainer import CustomTrainer

    
ultralytics.yolo.engine.model.TASK_MAP['detect'][1] = CustomTrainer

model = YOLO('./model.yaml', task = 'detect')     # Build from YAML
model = model.load('yolov8n.pt') # Transfer weights

# Train the model
model.train(
    data='./dataset.yaml',
    epochs=100,
    imgsz=640,
    batch = 128,
    device= [0,1],
    amp = True
)


# Evaluate the model's performance on the validation set
results = model.val()





# Perform object detection on an image using the model

example = './datasets/ssdd/valid/images/000012_jpg.rf.55a194724e0e7923d38b81b3af595f59.jpg'
results = model.predict(source = example, imgsz=640, conf = 0.50)

for idx in range(len(results)):
  plt.imshow(results[idx].plot())
  plt.savefig('output.png')
  plt.show()

  orig_img = results[idx].orig_img
  plt.imshow(orig_img)
  plt.savefig('input.png')
  plt.show()

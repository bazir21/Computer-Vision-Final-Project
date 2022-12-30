import cv2
import os
import torch

model = torch.hub.load('../../yolov5', 'custom', path='../../yolov5/runs/train/exp2/weights/best.pt', source='local')
dir = "../data/images"

# images = []
# for filename in os.listdir(dir):
#     img = cv2.imread(os.path.join(dir, filename))
#     images.append(img)

images = cv2.imread(os.path.join(dir, "2.png"))

results = model(images)
print(results.pandas().xyxy[0])
# results.show()

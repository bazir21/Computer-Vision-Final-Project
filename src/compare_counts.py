import os

box_dir = "../data/bounding_box"
yolo_dir = "../../yolov5/runs/detect/exp13/labels"

accuracy_count = 0
yolo_strawberries = 0
actual_strawberries = 0
no_files = len(os.listdir(box_dir))

for filename in os.listdir(box_dir):
    num_lines = sum(1 for _ in open(os.path.join(box_dir, filename)))
    yolo_num_lines = sum(1 for _ in open(os.path.join(yolo_dir, filename)))

    actual_strawberries += num_lines
    yolo_strawberries += yolo_num_lines

    if num_lines == yolo_num_lines:
        accuracy_count += 1


print("yolo strawberries = ", yolo_strawberries)
print("actual strawberries = ", actual_strawberries)

# print(yolo_strawberries / actual_strawberries)
print(accuracy_count / no_files)

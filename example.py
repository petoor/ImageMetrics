import numpy as np 
from PIL import Image
from metrics.binary_metrics import BinaryImageMetrics

base_img = np.uint8(Image.open("./test_images/base.png").convert("L"))
expanded_img = np.uint8(Image.open("./test_images/expanded.png").convert("L"))
missing_img = np.uint8(Image.open("./test_images/missing.png").convert("L"))
edge = np.uint8(Image.open("./test_images/edge.png").convert("L"))
empty = np.zeros_like(base_img)
full = np.ones_like(base_img)

metrics = BinaryImageMetrics(base_img, expanded_img)
count = metrics.count_ratio()
print(f"Count ratio is: {count}")

f1 = metrics.f1()
print(f"F1 score is : {f1}")
f1_obj = metrics.f1(obj=True)
print(f"F1 object score is: {f1_obj}")

iou = metrics.iou()
print(f"IoU score is : {iou}")
iou_obj = metrics.iou(obj=True)
print(f"IoU object score is : {iou_obj}")

hausdorff_obj_distance = metrics.hausdorff_distance(obj=True)
print(f"Hausdroff object distance score is : {hausdorff_obj_distance}")

mcc = metrics.mcc()
print(f"MCC score is : {mcc}")

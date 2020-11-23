import numpy as np 
from PIL import Image
from metrics.binary_metrics import BinaryImageMetrics

base_img = np.uint8(Image.open("./test_images/base.png").convert("L"))
expanded_img = np.uint8(Image.open("./test_images/expanded.png").convert("L"))
missing_img = np.uint8(Image.open("./test_images/missing.png").convert("L"))
edge = np.uint8(Image.open("./test_images/edge.png").convert("L"))
empty = np.zeros_like(base_img)
full = np.ones_like(base_img)

metrics = BinaryImageMetrics(missing_img, expanded_img)
count = metrics.get_count()
print(f"Count ratio is: {count}")
f1 = metrics.get_f1()
print(f"F1 score is : {f1}")
f1_obj = metrics.get_f1_obj()
print(f"F1 object score is : {f1_obj}")
hausdorff_obj_distance = metrics.get_hausdorff_obj_distance()
print(f"Hausdroff object distance score is : {hausdorff_obj_distance}")

import numpy as np 
from PIL import Image
from metrics.bbox_metric import BBoxMetrics

base_img = np.uint8(Image.open("./test_images/base.png").convert("L"))
expanded_img = np.uint8(Image.open("./test_images/expanded.png").convert("L"))
missing_img = np.uint8(Image.open("./test_images/missing.png").convert("L"))
edge = np.uint8(Image.open("./test_images/edge.png").convert("L"))
empty = np.zeros_like(base_img)
full = np.ones_like(base_img)

metrics = BBoxMetrics(y_true=base_img, y_pred=empty)
print(metrics.cm())


import cv2
import numpy as np
from IPython.display import Image, display

def draw_boxes(img_path, ann_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(ann_path, 'r') as f:
        anns = f.readlines()

    for ann in anns:
        cid, cx, cy, bw, bh = map(float, ann.strip().split())

        x = int((cx - bw / 2) * w)
        y = int((cy - bh / 2) * h)
        width = int(bw * w)
        height = int(bh * h)

        color = (0, 255, 0)  # Change the color if needed
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
        label = f"Class {int(cid)}"

    cv2.imshow("Annotated Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img_path = '/content/drive/MyDrive/Pedestrian_dataset/dataset/augment/images/image_0.png'
ann_path = '/content/drive/MyDrive/Pedestrian_dataset/dataset/augment/labels/image_0.txt'

draw_boxes(img_path, ann_path)

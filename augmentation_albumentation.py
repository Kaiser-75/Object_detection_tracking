import cv2
import albumentations as A
import os

def read_yolo_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        labels = [list(map(float, line.strip().split())) for line in lines]
    return labels

def yolo_to_albumentations(labels, w, h):
    alb_labels = []
    for label in labels:
        x, y, width, height = label[1], label[2], label[3], label[4]
        x_min = max(0, int((x - width / 2.0) * w))
        y_min = max(0, int((y - height / 2.0) * h))
        x_max = min(int((x + width / 2.0) * w), w)
        y_max = min(int((y + height / 2.0) * h), h)
        alb_labels.append([x_min / w, y_min / h, x_max / w, y_max / h])
    return alb_labels

def albumentations_to_yolo(labels, w, h):
    yolo_labels = []
    for label in labels:
        x_min, y_min, x_max, y_max = label
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        yolo_labels.append([0, center_x, center_y, bbox_width, bbox_height])  # Assuming class_id is 0
    return yolo_labels

def save_yolo_labels(label_path, yolo_labels):
    with open(label_path, 'w') as f:
        for label in yolo_labels:
            f.write(' '.join(map(str, label)) + '\n')

def visualize_bboxes(image, bboxes):
    for bbox in bboxes:
        x, y, w, h = bbox
        x_min, y_min, x_max, y_max = int(x * image.shape[1]), int(y * image.shape[0]), int((x + w) * image.shape[1]), int((y + h) * image.shape[0])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return image

input_img_folder = '/content/drive/MyDrive/Pedestrian_dataset/images/train'
input_lbl_folder = '/content/drive/MyDrive/Pedestrian_dataset/labels/train'
output_img_dir = '/content/drive/MyDrive/Pedestrian_dataset/dataset/augment/images'
output_lbl_dir = '/content/drive/MyDrive/Pedestrian_dataset/dataset/augment/labels'
img_files = [f for f in os.listdir(input_img_folder) if f.endswith('.png')]

transform = A.Compose([
    A.Rotate(limit=45, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomCrop(width=400, height=400, p=0.5),
    A.GaussNoise(p=0.2),
], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_id']))

for img_file in img_files:
    img_path = os.path.join(input_img_folder, img_file)
    lbl_path = os.path.join(input_lbl_folder, img_file.replace('.png', '.txt'))
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_lbls = read_yolo_labels(lbl_path)
    img_h, img_w, _ = orig_img.shape
    alb_lbls = yolo_to_albumentations(orig_lbls, img_w, img_h)
    
    for i in range(5):
        augmented = transform(image=orig_img.copy(), bboxes=alb_lbls, category_id=[1] * len(alb_lbls))
        aug_img = augmented['image']
        aug_lbls = augmented['bboxes']
        yolo_aug_lbls = albumentations_to_yolo(aug_lbls, img_w, img_h)
        save_yolo_labels(os.path.join(output_lbl_dir, f'{os.path.splitext(img_file)[0]}_augmented_labels_{i}.txt'), yolo_aug_lbls)
        cv2.imwrite(os.path.join(output_img_dir, f'{os.path.splitext(img_file)[0]}_augmented_image_{i}.png'), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

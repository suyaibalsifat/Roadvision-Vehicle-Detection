'''
### 📄 **`train.py` (Final Pipeline)**
This is the cleaned version that reproduces your best submission (`submission_raw_60.csv`).  
You can adapt it to include your other experiments if you wish.
python'''

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# =====================================================================
# 1. Dataset Preparation (Stratified Split)
# =====================================================================
def prepare_dataset(zip_path='/content/road-vision.zip'):
    import zipfile, shutil
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall('/content/')
    base = '/content/RoadVision_DUET'
    train_img = os.path.join(base, 'train/images')
    train_csv = os.path.join(base, 'train/train.csv')
    test_dir = os.path.join(base, 'test/images')
    df = pd.read_csv(train_csv)
    df['base_id'] = df['image_id'].apply(lambda x: os.path.splitext(x)[0])
    # Stratify rare classes
    img_class_counts = df.groupby(['base_id', 'class_id']).size().unstack(fill_value=0)
    rare = [2, 8, 11, 12]
    val_imgs = set()
    for cls in rare:
        if cls in img_class_counts.columns:
            cls_imgs = img_class_counts[img_class_counts[cls] > 0].index.difference(val_imgs)
            if len(cls_imgs) > 1:
                _, v = train_test_split(cls_imgs, test_size=0.15, random_state=42)
                val_imgs.update(v)
            elif len(cls_imgs) == 1:
                val_imgs.update(cls_imgs)
    remaining = img_class_counts.index.difference(val_imgs)
    _, std_val = train_test_split(remaining, test_size=0.08, random_state=42)
    val_imgs.update(std_val)
    train_imgs = img_class_counts.index.difference(val_imgs)
    # Create YOLO dataset
    out = '/content/yolo_dataset'
    for s in ['train', 'val']:
        os.makedirs(os.path.join(out, 'images', s), exist_ok=True)
        os.makedirs(os.path.join(out, 'labels', s), exist_ok=True)
    for split, ids in [('train', train_imgs), ('val', val_imgs)]:
        for base_id in ids:
            src_img = os.path.join(train_img, f'{base_id}.jpg')
            dst_img = os.path.join(out, 'images', split, f'{base_id}.jpg')
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
                sub = df[df['base_id'] == base_id]
                with open(os.path.join(out, 'labels', split, f'{base_id}.txt'), 'w') as f:
                    for _, row in sub.iterrows():
                        f.write(f"{int(row['class_id'])} {row['x_center']:.6f} {row['y_center']:.6f} {row['width']:.6f} {row['height']:.6f}\n")
    # Write data.yaml
    yaml = f"""
path: {out}
train: images/train
val: images/val
nc: 13
names: ['Rickshaw','Motorcycle','Tempu','Sedan Car','Pickup','Microbus','Mini Bus','Mini Truck','Agro Use','Medium Truck','Large Bus','Heavy Truck','Trailer']
"""
    with open('/content/data.yaml', 'w') as f:
        f.write(yaml)
    return '/content/data.yaml', test_dir

# =====================================================================
# 2. Training (60 epochs)
# =====================================================================
def train_model(data_yaml):
    model = YOLO('yolo11x.pt')
    model.train(
        data=data_yaml,
        epochs=60,                # 40 + 20 fine-tune
        imgsz=1024,
        batch=2,
        optimizer='AdamW',
        lr0=0.01,
        cos_lr=True,
        warmup_epochs=3,
        mosaic=1.0,
        mixup=0.35,
        copy_paste=0.3,
        scale=0.8,
        degrees=15.0,
        translate=0.1,
        shear=3.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.1,
        box=7.5,
        cls=2.0,
        dfl=1.5,
        device=0,
        patience=20,
        save_period=10,
    )
    weights = glob.glob('/content/runs/detect/train*/weights/best.pt')[-1]
    return weights

# =====================================================================
# 3. Inference – Raw NMS (best submission)
# =====================================================================
def predict_raw(model_path, test_dir):
    model = YOLO(model_path)
    test_imgs = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    rows = []
    for img_name in test_imgs:
        base = os.path.splitext(img_name)[0]
        path = os.path.join(test_dir, img_name)
        res = model(path, imgsz=1024, augment=True, conf=0.01, verbose=False)[0]
        pred_str = ""
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxyn.cpu().numpy().tolist()
            scores = res.boxes.conf.cpu().numpy().tolist()
            labels = res.boxes.cls.cpu().numpy().astype(int).tolist()
            # Simple NMS (class-wise) – reproduce raw submission
            import torch
            from torchvision.ops import nms
            keep = []
            labels_t = torch.tensor(labels)
            boxes_t = torch.tensor(boxes)
            scores_t = torch.tensor(scores)
            for l in torch.unique(labels_t):
                mask = labels_t == l
                if mask.sum() == 0:
                    continue
                b = boxes_t[mask]
                s = scores_t[mask]
                idx = nms(b, s, 0.50)
                keep.extend(mask.nonzero()[idx].flatten().tolist())
            # Build string
            tokens = []
            for i in keep:
                x1,y1,x2,y2 = boxes[i]
                w,h = x2-x1, y2-y1
                if w > 0.005 and h > 0.005:
                    cx,cy = (x1+x2)/2, (y1+y2)/2
                    tokens.extend([int(labels[i]), round(float(scores[i]),4), round(cx,6), round(cy,6), round(w,6), round(h,6)])
            pred_str = " ".join(map(str, tokens))
        rows.append({"image_id": base, "PredictionString": pred_str})
    df = pd.DataFrame(rows)
    df.to_csv('/content/submission_final.csv', index=False)
    print("✅ Submission saved: /content/submission_final.csv")

if __name__ == "__main__":
    data_yaml, test_dir = prepare_dataset()
    weights = train_model(data_yaml)
    predict_raw(weights, test_dir)

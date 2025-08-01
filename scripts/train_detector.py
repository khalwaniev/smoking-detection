#!/usr/bin/env python
"""
Train Faster R-CNN (ResNet50 FPN) on your cigarette/vape/firecracker dataset.
Run this on Colab with GPU.
"""
import argparse, os, json, random, time
from pathlib import Path
import numpy as np
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import cv2

CLASSES = ["__background__", "cigarette", "vape", "firecracker"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/smoking_detector", type=str)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--batch", default=4, type=int)
    ap.add_argument("--lr", default=2e-4, type=float)
    ap.add_argument("--out", default="models/object_detector", type=str)
    return ap.parse_args()

# -------- dataset ----------
class COCODet(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, ann_path, transforms=None):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_path)
        self.ids = list(self.coco.imgs.keys())
        self.imgs_dir = Path(imgs_dir)
        self.tfms = transforms

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = self.imgs_dir / coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        boxes, labels = [], []
        for a in anns:
            x,y,w0,h0 = a['bbox']
            boxes.append([x, y, x+w0, y+h0])
            labels.append(a['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0]),
            "iscrowd": torch.zeros((len(anns),), dtype=torch.int64)
        }
        if self.tfms:
            img = self.tfms(img)
        return img, target
    def __len__(self): return len(self.ids)

def img_tfms(img):
    img = cv2.resize(img, (640,640))
    img = torch.as_tensor(img/255., dtype=torch.float32).permute(2,0,1)
    return img

# -------- train ----------
def collate_fn(batch): return tuple(zip(*batch))

def main():
    args = parse_args()
    root = Path(args.root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = COCODet(root/"train", root/"annotations/instances_train.json", img_tfms)
    val_ds   = COCODet(root/"val",   root/"annotations/instances_val.json",   img_tfms)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_fn, num_workers=4)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          collate_fn=collate_fn, num_workers=2)

    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights,
                 num_classes=len(CLASSES))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr)
    lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)

    best_map = 0
    for epoch in range(args.epochs):
        model.train(); tot=0
        for imgs, targets in train_ld:
            imgs = [i.to(device) for i in imgs]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            losses = sum(loss_dict.values())
            optim.zero_grad(); losses.backward(); optim.step()
            tot += losses.item()
        lr_sched.step()
        map05 = evaluate(val_ld, model, device)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={tot/len(train_ld):.3f}  mAP@0.5={map05:.3f}")
        if map05 > best_map:
            best_map = map05
            out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir/"best.pth")
    print("Done. Best mAP@0.5:", best_map)

# quick evaluation (IoU>0.5, per image)
def evaluate(loader, model, device):
    model.eval(); hits=0; totals=0
    with torch.no_grad():
        for imgs, targets in loader:
            img = imgs[0].to(device)
            target = targets[0]
            pred = model([img])[0]
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels= target['labels'].cpu().numpy()
            for b,l in zip(gt_boxes, gt_labels):
                totals += 1
                for pb,pl,ps in zip(pred['boxes'].cpu().numpy(),
                                    pred['labels'].cpu().numpy(),
                                    pred['scores'].cpu().numpy()):
                    if ps<0.5 or pl!=l: continue
                    iou = box_iou(pb, b)
                    if iou>0.5: hits +=1; break
    return hits/max(totals,1)

def box_iou(a,b):
    # boxes in x1,y1,x2,y2
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(ix2-ix1,0)*max(iy2-iy1,0)
    if inter==0: return 0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter/(area_a+area_b-inter)

if __name__ == "__main__":
    main()

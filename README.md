# RoadVision – Vehicle Detection in Bangladesh Highway Surveillance

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/roadvision-duet)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-green)](https://github.com/ultralytics/ultralytics)

## 🏆 Competition Results
- **Public LB:** 0.60597  
- **Private LB:** 0.58099  
- **Top 20% finish** (estimated)

## 📌 Overview
This repository contains my solution for the **RoadVision** competition, where the goal is to detect and classify 13 vehicle types in real surveillance imagery from Bangladesh roads. The dataset has ~810 training images with 13 classes, including rare categories like *Mini Truck* and *Medium Truck*.

## 🔍 Approach
- **Base Model:** YOLO11x (pretrained on COCO)
- **Training Strategy:**
  - 60 epochs fine‑tuning with AdamW, cosine annealing
  - Stratified validation to preserve rare classes
  - Aggressive augmentations: mosaic, mixup, copy‑paste, HSV, rotation, scaling
- **Inference:** Multi‑scale TTA (1024px + 1280px) with Weighted Boxes Fusion (WBF)
- **Post‑processing:** Class‑specific confidence thresholds and NMS

## 📊 Key Results
| Model | Public LB | Private LB |
|-------|-----------|------------|
| YOLO11x (60 epochs, raw NMS) | **0.60597** | **0.58099** |
| YOLO11x + WBF (skip=0.08) | 0.57254 | 0.55311 |
| Blended ensemble | 0.58212 | 0.53156 |

The **raw NMS submission** (no WBF) performed best on the private LB, confirming that simpler post‑processing can generalise better.

## 🚀 How to Reproduce
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

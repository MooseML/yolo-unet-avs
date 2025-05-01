# YOLO-UNet-AVS

A project for real-time object detection and semantic segmentation of road scenes, using **YOLOv2** and **U-Net** for autonomous vehicle perception tasks.

## Project Overview

This project uses:
- **YOLOv2** for fast object detection (cars, pedestrians, traffic lights, etc.)
- **U-Net** for pixel-accurate road scene segmentation
- Python code and Jupyter notebooks for model loading, evaluation, and visualization

Pretrained models are used for object detection. Segmentation is trained from scratch on a subset of the CARLA dataset.

## Models

| Task           | Model   | Description                             |
|----------------|---------|-----------------------------------------|
| Object Detection | YOLOv2  | Pretrained on COCO, detects 80 classes |
| Segmentation     | U-Net   | Custom-trained on road scene masks     |


## Folder Structure

```
yolo-unet-avs/
├── notebooks/                # Main Jupyter notebooks
├── src/                     # Modular code for detection/segmentation
│   ├── detection/           # YOLO model loading + prediction
│   └── segmentation/        # U-Net model + data pipeline
├── data/                    # Input data and masks
│   ├── segmentation/
│   └── detection/
├── assets/                  # (Optional) Videos and media
├── outputs/                 # (Optional) Saved predictions
└── README.md
```

## Quickstart

1. Clone the repo:
   ```bash
   git clone https://github.com/MooseML/yolo-unet-avs.git
   cd yolo-unet-avs
   ```

2. Install dependencies (`conda` or `pip`)

3. Run:
   - `notebooks/unet_segmentation.ipynb` for segmentation
   - `notebooks/yolo_detection.ipynb` for object detection


## Working List of Projects To Explore

- [x] Detect objects in images
- [x] Segment road scenes with U-Net
- [x] Run YOLO on entire sample image folder
- [x] Video frame-by-frame detection (OpenCV)
- [ ] Real-time webcam inference
- [ ] Dashboard with object counts
- [ ] Inference benchmarking


## References

- [YOLOv2 Paper](https://arxiv.org/abs/1612.08242)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Drive.ai Sample Dataset (CC BY 4.0)](https://github.com/udacity/self-driving-car)

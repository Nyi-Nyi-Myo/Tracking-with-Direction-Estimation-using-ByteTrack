![🤖_Tracking_with_Direction_Estimation  _ _using_ByteTrack](https://github.com/user-attachments/assets/efe453e8-3780-433c-8730-c90835596c80)

[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
![Static Badge](https://img.shields.io/badge/Direction-Tracking-cyan)
![Static Badge](https://img.shields.io/badge/YOLOv8-8A2BE2)

Surgical instruments Tracking using modified ByteTrack Algorithm adapted with Direction Estimation module with the aim of keeping the correct IDs of the two instruments when crossing each other to differentiate between them. 

## Methodology
- Direction estimation using **OpenCV** and **NumPy**
- Instance Segmentation Model
- PyTorch, IPython.display, torch-2.6 bypass weights_only
- Google Colab, Google Drive

## How to apply
1. Main implementation for direction estimation in `direction tracking.py`
2. Put `byte_tracker.py` and `track.py` in YOLOv8
3. Check `yolov8_SI_track.ipynb` for running on Colab

## Results
https://github.com/user-attachments/assets/9480e49f-de69-49eb-834d-f3e6e9a2a1f7




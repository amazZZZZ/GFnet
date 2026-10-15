# GFnet

underwater object detection model built on Ultralytics YOLO.

## Installation

```bash
pip install ultralytics
```

## Usage

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/UOD/GFnet.yaml")
model.train(data="ultralytics/cfg/datasets/URPC.yaml", epochs=300, batch=16)
```

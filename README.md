# UODMamba

Mamba-based underwater object detection model built on Ultralytics YOLO.

## Installation

```bash
pip install seaborn thop timm einops
cd selective_scan && pip install . && cd ..
pip install ultralytics
```

## Usage

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/UOD/UODMamba.yaml")
model.train(data="ultralytics/cfg/datasets/DUO.yaml", epochs=300, batch=16)
```

## Datasets

- [DUO](https://github.com/chongweiliu/DUO)
- [RUOD](https://github.com/dlut-dimt/RUOD)

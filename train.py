from ultralytics import YOLO
from ultralytics import RTDETR
# 加载预训练模型
model = YOLO("ultralytics\cfg\models\UOD\UODMamba.yaml")
if __name__ == '__main__': 
    # Use the model
    results = model.train(data='ultralytics/cfg/datasets/DUO.yaml', epochs=300, batch=16)  # 训练模型
    # results = model.train(data='ultralytics/cfg/datasets/RUOD.yaml', epochs=300, batch=16)  # 训练模型

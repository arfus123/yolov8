from ultralytics import YOLO
# 推理
model = YOLO('./best.pt')
results = model(task='detect', mode='predict', source='datasets/images/Image1.jpg', line_width=3, show=True, save=True, device='cpu')
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")
    model.train(data = "my-yolo.yaml", epochs = 1000, imgsz = 640, batch = 200)
    

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.yaml")
    model.train(data = "my-yolo.yaml", epochs = 10000, imgsz = 640, batch = 100)
    

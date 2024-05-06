from ultralytics import YOLO

def main():
  model = YOLO('yolov8n.pt')

def predict_image(model: YOLO, image_path:str):
  yolo_results = model([image_path])



if __name__ == '__main__':
  main(path)
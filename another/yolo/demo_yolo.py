from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# results = model(['im1.jpg', 'im2.jpg'])  # return a list of Results objects
results = model(["./images/Pojazdy-na-ulicy.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

    for i in range(len(boxes)):
        # Pobieramy informacje o każdym wykrytym obiekcie
        box = boxes[i]
        mask = masks[i]
        keypoint = keypoints[i]
        prob = probs[i]
        
        # Tworzymy słownik zawierający informacje o obiekcie
        object_info = {
            'Box': box,
            'Mask': mask,
            'Keypoints': keypoint,
            'Probability': prob
        }
        
        # Dodajemy słownik do listy wykrytych obiektów
        detected_objects.append(object_info)

    result.show()  # display to screen
    result.save(filename='./images/result.jpg')  # save to disk
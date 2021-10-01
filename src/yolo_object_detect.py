import cv2
import numpy as np
import os
import time
from pprint import pprint

# Config
cfg = {}
cfg['yolo'] = "src\Resources\yolo"
cfg['confidence'] = 0.5
cfg['threshold'] = 0.5


labelsPath = os.path.sep.join([cfg["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split('\n')


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join([cfg["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([cfg["yolo"], "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Main function
def YOLO_Op():
    print("""
        Welcome to You-Only-Look-Once
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    """)
    path = input("Enter the image path: ")
    print(path)
    cfg['input'] = path.replace("\\", "/")
    if(allowed_file(cfg['input'])):

        cfg['output'] = "Output folder/output." + cfg['input'].split(".")[-1]

        image = cv2.imread(cfg["input"])
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layersOutputs = net.forward(ln)
        end = time.time()

        print(f"[INFO] Took {end - start} seconds...")
        boxes = []
        confidences = []
        classIDs = []

        for output in layersOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > cfg["confidence"]:
                    box = detection[0:4]*np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(
                boxes, confidences, cfg["confidence"], cfg["threshold"])

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                    text = f"{LABELS[classIDs[i]]} {round(confidences[i], 3)}"
                    cv2.putText(image, text, (x, y-5),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

            print("[INFO] the following classes were predicted...")
            classes = [LABELS[d] for d in classIDs]
            pprint(dict(zip(classes, confidences)))
            cv2.imwrite(f"{cfg['output']}", image)
            print("\n\n")
            print(f"Image is saved at this location - {cfg['output']}")
    else:
        print("[INFO] Image is not allowed")

    return

import cv2
import numpy as np
from speech import speak

config_file = r'config_files//YOLO_config.cfg'
YOLO_model = r'config_files//YOLOv3.weights'

model = cv2.dnn.readNet(YOLO_model, config_file)

classLabels = []
file_name = 'label.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# print(len(classLabels))

cap = cv2.VideoCapture(0)

count = 0
while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    model.setInput(blob)

    output_layers_names = model.getUnconnectedOutLayersNames()
    layerOutputs = model.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.5)

    font = cv2.FONT_HERSHEY_COMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))


    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]
            label = str(classLabels[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y), font, 1, (0, 255, 0), 1)
            if(label != 'person'):
                speak(label)
                count = 0
            elif(count > 5):
                speak('bring object in the frame')
            count += 1


    cv2.imshow('Image', img)
    # print(classLabels[int(ClassIndex-1)])
    #cap.release()
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()

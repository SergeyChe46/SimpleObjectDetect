import cv2
import numpy as np
from argparse import ArgumentParser
from tensorflow.keras import applications
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression

ap = ArgumentParser()
ap.add_argument('-i', '--image')
ap.add_argument('-f', '--filter', type=str)
args = vars(ap.parse_args())


def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()    
    return rects


label_filters = args['filter']
if label_filters is not None:
    label_filters = label_filters.lower().split(',')


model = applications.ResNet50(weights='imagenet')
image = cv2.imread(args['image'])
(H, W) = image.shape[:2]

rects = selective_search(image)

proposals = []
boxes = []

for (x, y, w, h) in rects:
    if w / float(W) < 0.1 or h / float(H) < 0.1:
        continue

    roi = image[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    proposals.append(roi)
    boxes.append((x, y, w, h))

proposals = np.array(proposals)
preds = model.predict(proposals)
preds = applications.decode_predictions(preds)

labels = {}


for (i, p) in enumerate(preds):
    (imagenet, label, prob) = p[0]

    if label_filters is not None and label not in label_filters:
        continue

    (x, y, w, h) = boxes[i]
    box = (x, y, x+w, y+h)

    L = labels.get(label, [])
    L.append((box, prob))
    labels[label] = L

    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)
    for (startX, startY, endX, endY) in boxes:
        clone = image.copy()
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, (label, proba), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow(clone)
    cv2.waitKey(0)
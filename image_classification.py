"""Opencv classifies images with dnn imagenet."""

import numpy as np
import cv2
from logzero import logger


class_path = 'models/synset_words.txt'
prototxt_path = 'models/bvlc_googlenet.prototxt'
model_path = 'models/bvlc_googlenet.caffemodel'

# loading the labels
logger.info('Loading labels...')
rows = open(class_path).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# loading model
logger.info('loading model...')
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# loading the demo img
image_path = 'imgs/img_1.jpg'
image = cv2.imread(image_path)
resized = cv2.resize(image, (244, 244))
blob = cv2.dnn.blobFromImage(resized, 1, (244, 244), (104, 117, 123))
print("First Blob: {}".format(blob.shape))

# setting the input to the net
net.setInput(blob)
preds = net.forward()

# sorting the probabilities
idx = np.argsort(preds[0])[::-1][0]
text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
cv2.putText(image, text, (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)

# save the image
logger.info('Saving image...')
cv2.imwrite('output/demo1.jpg', image)

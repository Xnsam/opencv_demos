"""Identify face landmarks in an image."""

# facial regions - mouth, right eyebrow, left eyebrow, right eye, left eye, nose, jaw

import numpy as np
import imutils
import dlib
import cv2
from logzero import logger


def rect_to_bb(rect):
    # convert the rect predicted by dlib and convert it to (x, y, w, h) format
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x ,y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x ,y) - cordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them to a 2 - tuple of (x , y) - cordinate
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

model_path = 'models/shape_predictor_68_face_landmarks.dat'
img_url = 'imgs/img_1.jpg'

logger.info('Loading the model...')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

logger.info('Loading the images...')
image = cv2.imread(img_url)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

logger.info('detecting faces...')
rects = detector(gray, 1)

logger.info('looping over the face detections...')
for (i, rect) in enumerate(rects):
    # convert the facial landmarks to numpy array
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    # convert dlib's rectangle to a openCV-style bounding box
    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

logger.info("saving the image...")
cv2.imwrite("output/landmarks_output.png", image)

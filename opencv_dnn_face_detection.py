"""Opencv dnn face detection in python."""

import numpy as np
import cv2
from logzero import logger

prototxt_path = 'models/deploy.prototxt.txt'
model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
img_path = 'imgs/img_1.jpg'
suggested_confidence = 0.7

# load the model from disk
logger.info('Loading model...')
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# load - construct blob - resize - normalize - image
image = cv2.imread(img_path)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# apply face detection
logger.info("computing bject detections...")
net.setInput(blob)
detections = net.forward()

logger.info('Drawing box...')
# box the detections
for i in range(0, detections.shape[2]):
    # extracting the confidence
    confidence = detections[0, 0, i, 2]
    # filter weak detections
    if confidence > suggested_confidence:
        # computing the x -y co odrinates for bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # drawing box with confidence
        text = '{:.2f}%'.format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 5)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

logger.info('Saving Image...')
cv2.imwrite("output/img_1.jpg", image)

# detection from video

# from imutils.video import VideoStream
# import numpy as np
# import imutils
# import time
# import cv2
# from logzero import logger

# model_path = ''
# prototxt_path = ''

# logger.info('Loading models...')
# net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# logger.info('starting the video stream...')
# vs = VideoStream(src=0).start()
# # vs = FileVideoStream(src=0).start()
# time.sleep(2.0)
# while True:
#     # reading the frame - resizing the frame
#     frame = vs.read()
#     frame = imutils.resize(frame, width=400)
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     # pass the blob
#     net.setInput(blob)
#     detections = net.forward()
#     # drawing boxes
#     logger.info('drawing box...')
#     for i in range(0, detections.shape[2]):
#         # extracting the confidence
#         confidence = detections[0, 0, i, 2]
#         # filter out weak detections
#         if confidence < suggested_confidence:
#             continue
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")
#         #draw confidence with box
#         text = "{:.2f}%".format(confidence * 100)
#         y = startY - 10 if startY - 10 > 10 else startY + 10
#         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
#         cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#         # show the output frame
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF
#         # exit the loop
#         if key = ord("q"):
#             break

# cv2.destroyAllWindows()
# vs.stop()

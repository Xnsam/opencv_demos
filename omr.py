"""Optical mark recognition."""

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
from logzero import logger

answer_key = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
img_path = 'imgs/omr_img.png'

# load the image , convert it to grayscale, blur it, slightly, find edges
logger.info('Loading image...')
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# find the contours in the edge map
logger.info('Finding contours...')
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

docCnt = None

# ensure that at least one contour is found
logger.info('Finding contours with four points...')
if len(cnts) > 0:
    # sorting the contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if contour has 4 points
        if len(approx) == 4:
            docCnt = approx
            break

print(docCnt)
# apply the four point perspective transform
logger.info('applying four point perspective...')
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# save warped image
cv2.imwrite("warped_omr.png", warped)

# binarize the warped piece of paper
logger.info('binarize the warped piece of paper...')
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# finding contours in threshold images
logger.info('finding contours to the threshold images...')
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []

# loop over the contours
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # in order to label the contour
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

logger.info('sorting the questions...')
# sort the question contours top-to-bottom, then initialize the total number of correct answers
quesitonCnts = contours.sort_contours(questionCnts, method='top-to-bottom')[0]
correct = 0

# sorting the answers to the questions
logger.info('Matching the questions and answer...')
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # sort the contours for the current question from left to right
    cnts = contours.sort_contours(questionCnts[i: i + 5])[0]
    bubbled = None
    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # apply the mask to threshold image, then count the number of non-zero pixels
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        # if current total has a larger number
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
            # initialize the contour color and the index of the correct answer
            color = (0, 0, 255)
            k = answer_key[q]

            # check to see if the bubbled answer is correct
            if k == bubbled[1]:
                color = (0, 255, 0)
                correct += 1

logger.info('Drawing contours...')
cv2.drawContours(paper, [cnts[k]], -1, color, 3)
score = (correct / 5.0) * 100
print("score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# save the image
logger.info('saving the result...')
cv2.imwrite('output/omr_result.png', paper)

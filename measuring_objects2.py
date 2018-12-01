"""Measure the objects in image using opencv."""


# size of card width = 3.9 inch , height = 2.36
# pixels per metric = object_width / know_width.

import cv2
import imutils
from scipy.spatial import distance as dist
import numpy as np
from imutils import perspective
from imutils import contours
from logzero import logger


def mid_point(pt_a, pt_b):
    """Calculating mid point of the function."""
    return ((pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5)


logger.info('Loading image...')
img_url = 'imgs/img_9.jpg'
image = cv2.imread(img_url)
# image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (7, 7), 0)

# logger.info('detecting edge - dilating - eroding...')
# edged = cv2.Canny(gray, 50, 100)
# edged = cv2.dilate(edged, None, iterations=1)
# edged = cv2.erode(edged, None, iterations=1)
# cv2.imwrite('demos/edged.png', edged)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

cv2.imwrite('demos/gray.png', gray)
logger.info('thresholding - dilating - eroding...')
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# kernel = np.ones((5, 5), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=20)
thresh = cv2.erode(thresh, kernel, iterations=20)
cv2.imwrite('demos/thresh.png', thresh)
# edged = cv2.Canny(thresh, 50, 100)
# edged = cv2.dilate(edged, None, iterations=20)
# edged = cv2.erode(edged, None, iterations=20)
# cv2.imwrite('demos/edged.png', edged)

logger.info('Finding contours...')

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

logger.info('Sorting contours...')
(cnts, _) = contours.sort_contours(cnts)
pixels_per_metric = None

logger.info('checking the contours...')

for c in cnts:
    a = cv2.contourArea(c)
    if a < 200:
        continue
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 0, 255), 5)

    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    logger.info('Saving Images...')
    cv2.imwrite('demos/{}.png'.format(a), orig)

    (tl, tr, br, bl) = box
    logger.debug('calculating midpoints top-left -- top-right')
    (tltrx, tltry) = mid_point(tl, tr)
    logger.debug('calculating midpoints bottom-left -- bottom-right')
    (blbrx, blbry) = mid_point(bl, br)
    logger.debug('calculating midpoints top-left -- bottom-left')
    (tlblx, tlbly) = mid_point(tl, bl)
    logger.debug('calculating midpoints top-right -- bottom-right')
    (trbrx, trbry) = mid_point(tr, br)

    logger.info('Draing mid-points')
    cv2.circle(orig, (int(tltrx), int(tltry)), 5, (255, 0, 0,), -1)
    cv2.circle(orig, (int(blbrx), int(blbry)), 5, (255, 0, 0,), -1)
    cv2.circle(orig, (int(tlblx), int(tlbly)), 5, (255, 0, 0,), -1)
    cv2.circle(orig, (int(trbrx), int(trbry)), 5, (255, 0, 0,), -1)
    cv2.line(orig, (int(tltrx), int(tltry)), (int(blbrx), int(blbry)), (255, 0, 255), 2)
    cv2.line(orig, (int(tlblx), int(tlbly)), (int(trbrx), int(trbry)), (255, 0, 255), 2)

    logger.info('computing the euclidean distance between midpoints')

    d_a = dist.euclidean((tltrx, tltry), (blbrx, blbry))
    d_b = dist.euclidean((tlblx, tlbly), (trbrx, trbry))

    if pixels_per_metric is None:
        pixels_per_metric = d_b / 1.96

    logger.info('computing size of the object')
    dim_a = d_a / pixels_per_metric
    dim_b = d_b / pixels_per_metric
    cv2.putText(orig, "{:.1f}in".format(dim_a), (int(trbrx + 10), int(trbry)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dim_b), (int(tltrx - 15), int(tltry - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    logger.info('saving measured images')
    cv2.imwrite("demos/final_img_{}.png".format(a), orig)

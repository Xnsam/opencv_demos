"""Measure the sizes of width known objects."""

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from logzero import logger


# function for calculating the mid point
def mid_point(pt_a, pt_b):
    return ((pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5)


img_url = 'imgs/sz_obj.jpg'
width = 0.955

# image loading and preprocessing
logger.info("Loading objects...")
image = cv2.imread(img_url)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection
logger.info("performing edge detection...")
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# finding contours
logger.info("finding contours...")
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# sort the contours from left to right
logger.info("sorting contours...")
(cnts, _) = contours.sort_contours(cnts)
pixels_per_metric = None

# loop over the contours individually
logger.info("Looping over contours...")
for c in cnts:
    if cv2.contourArea(c) < 100:
        continue
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        (tl, tr, br, bl) = box
        (tltrx, tltry) = mid_point(tl, tr)
        (blbrx, blbry) = mid_point(bl, br)
        (tlblx, tlbly) = mid_point(tl, bl)
        (trbrx, trbry) = mid_point(tr, br)

        cv2.circle(orig, (int(tltrx), int(tltry)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrx), int(blbry)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblx), int(tlbly)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrx), int(trbry)), 5, (255, 0, 0), -1)

        cv2.line(orig, (int(tltrx), int(tltry)), (int(blbrx), int(blbry)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblx), int(tlbly)), (int(trbrx), int(trbry)), (255, 0, 255), 2)

        dA = dist.euclidean((tltrx, tltry), (blbrx, blbry))
        dB = dist.euclidean((tlblx, tlbly), (trbrx, trbry))

        if pixels_per_metric is None:
            pixels_per_metric = dB / width

        dimA = dA / pixels_per_metric
        dimB = dB / pixels_per_metric

        cv2.putText(orig, "{:.1f} in".format(dimA), (int(tltrx - 15), int(tltry - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f} in".format(dimB), (int(trbrx + 10), int(trbry)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        logger.info("saving objects...")
        cv2.imwrite("output/measure_img_{}.png".format(dimA), orig)

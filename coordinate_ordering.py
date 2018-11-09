"""How to co ordinate the order."""

# from __future__ import print_function
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from logzero import logger


def order_points_old(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# def order_points(pts):
#   # sorted the points based on their x-coordinates
#   xSorted = pts[np.argsort(pts[:, 0]), :]

#   # left most and right - most points from the sorted x - coordinate points
#   leftmost = xSorted[:2, :]
#   rightmost = xSorted[:2, :]

#   # leftmost coordinates according to y-coordinates
#   leftmost = leftmost[np.argsort(leftmost[:, 1]), :]
#   (tl, bl) = leftmost

#   # euclidean distance between the top-left and right-most points
#   # pythagorean theorem, the point with laget distance will be our bottom-right point
#   D = dst.cdist(tl[np.newaxis], rightmost, "euclidean")[0]
#   (br, tr) = rightmost[np.argsort(D)[::-1], :]
#   return np.array([tl, tr, br, bl], dtype="float32")

logger.info("Loading image")
# loading image
image_path = 'imgs/image3.jpeg'
image = cv2.imread(image_path, 0)
# converting to gray - blur
im = image.copy()
gray = cv2.GaussianBlur(im, (7, 7), 0)

# perform edge detection
logger.info('detecting edges...')

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# cv2.imwrite("output/coordinate.png", edged)
logger.info("finding contours...")
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# sort the points
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

# loop over contours individually
for (i, c) in enumerate(cnts):
    if cv2.contourArea(c) < 2000:
        continue

    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype='int')
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    # print("object # {}".format(i + 1))
    # print(box)
    rect = order_points_old(box)
    if -1 > 0:
        rect = perspective.order_points(box)
    print(rect.astype("int"))

    for ((x, y), color) in zip(rect, colors):
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    cv2.putText(image, "object # {}".format(i + 1), (int(rect[0][0] - 15), int(rect[0][1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

cv2.imwrite('output/contoured.png', image)


def sort_array(x):
    for i in range(0, len(x)):
        for j in range(i + 1, len(x)):
            if x[j] < x[i]:
                tmp = x[i]
                x[i] = x[j]
                x[j] = tmp

sort_array([2,4,6,8,3])


def insertion_sort(n, arr):
    num = arr[-1]
    for i in range(n-2, -1, -1):
        if arr[i] > num:
            arr[i+1] = arr[i]
            print(' '.join(str(j) for j in arr))
        else:
            arr[i+1] = num
            print(' '.join(str(j) for j in arr))
        arr[0] = num
        print(' '.join(str(j) for j in arr))

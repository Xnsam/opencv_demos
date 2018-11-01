"""A document scanner."""

from tranform import four_point_transform
from skimage.filters import threshold_local
import cv2
import imutils
from logzero import logger

img_path = 'imgs/1.jpg'
save_path = 'output/demo2.png'

# ============================= detect Edges
logger.info('Starting edge detection...')
# load image
logger.info('Loading image...')
image = cv2.imread(img_path)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# convert to grayscale - blur - find edges
logger.info('Processing image...')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# save the edge detected image
logger.info('saving edge detected image...')
cv2.imwrite('output/edge.png', edged)

# ============================== find contours
logger.info('finding contours...')
# find the largest contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over the contours
screenCnt = []
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if approximated contour has four points
    if len(approx) == 4:
        logger.debug('Found contour')
        screenCnt.append(approx)
        break

if len(screenCnt) > 0:
    # save the contour detected image
    logger.info('saving contour image')
    cv2.drawContours(image, [screenCnt[0]], -1, (0, 255, 0), 2)
    cv2.imwrite('output/cnt.png', image)

    # ============================== applying the perspective transform
    logger.info('starting perspective transform')
    # warping the image in four point transform
    warped = four_point_transform(orig, screenCnt[0].reshape(4, 2) * ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    # save the scanned images
    logger.info('saving perspective transform ')
    cv2.imwrite("output/scanned.png", imutils.resize(warped, height=650))
else:
    logger.info('No document found!')

"""Try template matching in python and opencv2."""

# ==================================================== template matching demo

# import numpy as np
# import cv2
# from logzero import logger
# import os

# img_url = 'imgs/simpsons.jpg'
# template_url = 'imgs/barts_face.jpg'

# logger.info('Loading images')
# img = cv2.imread(img_url)
# template = cv2.imread(template_url)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# w, h = template.shape[::-1]

# logger.info('started matching template')
# result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
# loc = np.where(result >= 0.9)

# logger.info('Drawing rectangle for the template')
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

# cv2.imwrite('imgs/matchedimg.png', img)


# ======================================================= template matching with scaling

import os
import cv2
import imutils
import numpy as np
from logzero import logger

img_url = 'imgs/comp1.jpg'
path = os.getcwd()
templates_path = path + '/imgs/pan_templates'
found = None

logger.info('Loading image')
img = cv2.imread(img_url)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
counter = 0

for file_path in sorted(os.listdir(templates_path)):
    templates_url = os.path.join(templates_path, file_path)
    counter += 1
    logger.info('Loading template {}'.format(counter))
    template = cv2.imread(templates_url)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # template = cv2.Canny(template, 10, 10)
    (th, tw) = template.shape[:2]

    logger.info('Searching for template...')
    for scale in np.linspace(0.1, 0.5, 50)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        if resized.shape[0] < th or resized.shape[1] < tw:
            break

        # edged = cv2.Canny(resized, 10, 10)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # clone = np.dstack([edged, edged, edged])
        # cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tw, maxLoc[1] + th), (0, 0, 255), 2)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tw) * r), int((maxLoc[1] + th) * r))
    im = img.copy()
    cv2.rectangle(im, (startX, startY), (endY, endY), (0, 255, 0), 2)
    cv2.imwrite('imgs/demos/{}.png'.format(counter), im)

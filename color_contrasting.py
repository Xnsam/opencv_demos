"""Color contrasting using python3."""

import cv2
from logzero import logger

logger.info('loading image')
img_url = 'imgs/img_2.jpg'
img = cv2.imread(img_url)

logger.info('converting image to Lab model')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)
# logger.info('saving images')
# cv2.imwrite('demos/lab_img.png', lab)
# cv2.imwrite('demos/l_channel.png', l)
# cv2.imwrite('demos/a_channel.png', a)
# cv2.imwrite('demos/b_channel.png', b)

logger.info('applying CLAHE to L channel')
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

# logger.info('saving images')
# cv2.imwrite('demos/clahe_img.png', cl)

logger.info('Merging the CLAHE enhanced L channel with a and b channel')
limg = cv2.merge((cl, a, b))
# cv2.imwrite('demos/merged_clahe_img.png', limg)

logger.info('Converting image from LAB color model to RGB model')
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imwrite('demos/final_img.png', final)

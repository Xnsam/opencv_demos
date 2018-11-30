"""Check if image is similar."""


import cv2
# import numpy as np
import imutils
# from skimage.measure import compare_ssim as ssim
# import matplotlib.pyplot as plt
from logzero import logger

img1_url = 'imgs/comp2.jpg'
img2_url = 'imgs/comp1.jpg'

# img1_url = 'imgs/converted_img.png'
# img2_url = 'imgs/demo1.png'


# def mse(img1, img2):
#     """Mean squared error."""
#     logger.info('Calculating MSE')
#     err = np.sum((img1.astype("float") - img2.astype("float"))**2)
#     err /= float(img1.shape[0] * img1.shape[1])
#     return err


def compare_images(img1, img2, title):
    """Compute the mean squared error and structural similarity."""
    logger.info('Started comparison')
    img1 = cv2.resize(img1, (350, 200))
    img2 = cv2.resize(img2, (350, 200))
    # img1 = imutils.resize(img1, width=940)
    # img2 = imutils.resize(img2, width=940)
    # m = mse(img1, img2)
    # logger.info('Calculating SSIM')
    # (score, diff) = ssim(img1, img2, full=True)
    # diff = (diff * 255).astype("uint8")
    # logger.info('saving difference image')
    # cv2.imwrite('imgs/comparediff.png', diff)
    # logger.debug('SSIM: {}'.format(score))

    logger.info('finding contours')
    thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imwrite('imgs/comparethresh.png', thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # counter = 0
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # ar = w / float(h)
        area = cv2.contourArea(c)
        # if ar > 1 and ar < 4:
        #     logger.debug(ar)
        #     counter += 1
        #     logger.info('cropping - saving image.')
        #     cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     out = img2[y: y + h, x: x + w]
        #     cv2.imwrite('imgs/img_{}.png'.format(counter), out)
        if area > 100:
            # logger.debug(area)
            # logger.debug('ar {}'.format(ar))
            out = img1[y: y + h, x: x + w]
            logger.info('cropping image with area {}'.format(area))
            cv2.imwrite('imgs/img_{}.png'.format(area), out)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    logger.info('saving images')
    cv2.imwrite('imgs/comparedemo1.png', img1)
    cv2.imwrite('imgs/comparedemo2.png', img2)

    # fig = plt.figure(title)
    # plt.suptitle("SSIM: %.2f" % (score))

    # fig.add_subplot(1, 2, 1)
    # plt.imshow(img1, cmap=plt.cm.gray)
    # plt.axis("off")

    # fig.add_subplot(1, 2, 2)
    # plt.imshow(img2, cmap=plt.cm.gray)
    # plt.axis("off")

    # plt.show()


def load_images(img1_url, img2_url):
    logger.info('Loading Images')
    img1 = cv2.imread(img1_url)
    img2 = cv2.imread(img2_url)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = cv2.GaussianBlur(img1, (9, 9), 0)
    img2 = cv2.GaussianBlur(img2, (9, 9), 0)

    # fig = plt.figure("Images")
    # images = ('img1', img1), ('img2', img2)

    # for (i, (name, image)) in enumerate(images):
    #     ax = fig.add_subplot(1, 3, i + 1)
    #     ax.set_title(name)
    #     plt.imshow(image, cmap=plt.cm.gray)
    #     plt.axis("off")

    # plt.show()
    compare_images(img1, img2, "img1 vs img2")

load_images(img1_url, img2_url)

"""
Methods

1. ssim
2. contours aspect ratio + template matching
3. contours area + template matching


"""

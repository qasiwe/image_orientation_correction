#  Document image orientation correction
#  This approach is based on text orientation

#  Assumption: Document image contains all text in same orientation

import cv2
import numpy as np
import os
import sys


def rotate(img, theta):  # rotate the image with given theta value
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)

    M = cv2.getRotationMatrix2D(image_center, theta, 1)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]

    # rotate original image to show transformation
    rotated = cv2.warpAffine(
        img, M, (bound_w, bound_h), borderValue=(255, 255, 255)
    )
    return rotated


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope_val = (y2-y1)/(x2-x1)
    theta = np.rad2deg(np.arctan(slope_val))
    return theta


def main(args):
    img = cv2.imread(args[0])

    if img is None:
        print("You specified wrong path to an image.")
        sys.exit()
    textImg = img.copy()

    small = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)

    #  find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    #  binarize the gradient image
    _, bw = cv2.threshold(
        grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    #  connect horizontally oriented regions
    #  kernel value (9,1) can be changed to improved the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(
        connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    mask = np.zeros(bw.shape, dtype=np.uint8)
    #  cumulative theta value
    cumulative_theta = 0
    #  number of detected text regions
    ct = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        #  fill the contour
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        #  ratio of non-zero pixels in the filled region
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        #  assume at least 45% of the area is filled if it contains text
        if r > 0.45 and w > 8 and h > 8:

            rect = cv2.minAreaRect(contours[idx])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(textImg, [box], 0, (0, 0, 255), 2)

            #  we can filter theta as outlier based on other theta values
            theta = slope(box[0][0], box[0][1], box[1][0], box[1][1])
            cumulative_theta += theta
            ct += 1

    #  find the average of all cumulative theta value
    orientation = cumulative_theta/ct
    print("Image orientation in degrees: ", orientation)

    finalImage = rotate(img, orientation)
    if len(args) > 1:
        path = args[1]
        cv2.imwrite(os.path.join(path, 'result.jpg'), finalImage)
    else:
        cv2.imwrite("result.jpg", finalImage)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        print('You have specified too many arguments')
        sys.exit()

    if len(sys.argv) < 2:
        print('You need to specify the path to your image')
        sys.exit()

    main(sys.argv[1:])

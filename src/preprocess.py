#!/usr/bin/env python3
import os
import cv2
import numpy
import argparse
from numba import jit
from matplotlib import pyplot as plt


# Take three channels and turn to greyscale image
@jit  # Use numba to make it run before the heat death of the universe
def merge_to_greyscale(r, g, b):
    height, width = r.shape
    out_image = numpy.zeros((height, width), dtype=numpy.uint8)
    for x in range(height):
        for y in range(width):
            grey_value = r[x][y] + g[x][y] + b[x][y]
            if grey_value > 255:
                grey_value = 255
            out_image[x][y] = grey_value

    return out_image


def preprocess(directory_name):
    images = os.listdir(directory_name)

    # Go through each image and process it with opencv
    for image_name in images:
        image_path = os.path.join(directory_name, image_name)
        bgr_img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2BGR)

        # Adaptive threshold
        # adaptive = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)
        r, g, b = cv2.split(rgb_img)

        ## Binarise each channel
        _, r_binary = cv2.threshold(r, 210, 255, cv2.THRESH_BINARY)
        _, g_binary = cv2.threshold(g, 210, 255, cv2.THRESH_BINARY)
        _, b_binary = cv2.threshold(b, 255, 255, cv2.THRESH_BINARY)

        ## Merge the binarised channels
        merged = merge_to_greyscale(r_binary, g_binary, b_binary)

        # contours, hierarchy = cv2.findContours(no_lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        ## if last digit of hierarchy, e.g. hierarchy[0][i][3] is -1 then it's parent, we should draw only if > threshold
        # no_dots = numpy.zeros_like(no_lines)
        ## Remove contours with area below threshold (removes random dots from the background)
        # filtered = []
        # for index, contour in enumerate(contours):
        #    area = cv2.contourArea(contour)
        #    # Always draw children, only draw parents if area large enough
        #    if hierarchy[0][index][3] != -1 or area > 40:
        #        filtered.append(contour)

        # cv2.drawContours(no_dots, filtered, -1, 255, -1)

        titles = ['Original', 'Red', 'Green', 'Binary Red', 'Binary Green', 'Merged']
        images = [rgb_img, r, g, r_binary, g_binary, merged]
        for i in range(len(images)):
            plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', help='Where to look for the image dataset', type=str)
    args = parser.parse_args()

    if args.image_dir is None:
        print("Please specify the path to the image directory")
        exit(1)

    preprocess(args.image_dir)


if __name__ == '__main__':
    main()

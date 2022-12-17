#!/usr/bin/env python3
import os
import cv2
import numpy
import argparse
from matplotlib import pyplot as plt


def convert_to_single_channel(img, idx):
    new = numpy.zeros_like(img)
    new[:, :, idx] = img[:, :, idx]
    return new


# Take three channels and turn to greyscale image
def merge_to_greyscale(r, g, b):
    height, width = r.shape
    out_image = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    for x in range(height):
        for y in range(width):
            grey_value = r[x][y] + g[x][y] + b[x][y]
            if grey_value > 255:
                grey_value = 255
            out_image[x][y][0] = grey_value
            out_image[x][y][1] = grey_value
            out_image[x][y][2] = grey_value

    return out_image


def preprocess(directory_name):
    images = os.listdir(directory_name)

    # Go through each image and process it with opencv
    for image_name in images:
        image_path = os.path.join(directory_name, image_name)
        bgr_img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2BGR)

        # Greyscale
        grey = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # Binarisation
        # _, binary = cv2.threshold(grey, 140, 255, cv2.THRESH_BINARY_INV)

        # Adaptive threshold
        # adaptive = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)

        img_onlyR = convert_to_single_channel(rgb_img, 0)
        img_onlyG = convert_to_single_channel(rgb_img, 1)
        img_onlyB = convert_to_single_channel(rgb_img, 2)

        # Binarise each channel
        _, r_binary = cv2.threshold(img_onlyR, 210, 255, cv2.THRESH_BINARY)
        _, g_binary = cv2.threshold(img_onlyG, 210, 255, cv2.THRESH_BINARY)
        _, b_binary = cv2.threshold(img_onlyB, 255, 255, cv2.THRESH_BINARY)

        # Merge the binarised channels
        merged = merge_to_greyscale(r_binary[:, :, 0], g_binary[:, :, 1], b_binary[:, :, 2])

        # Greyscale
        #grey = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)

        # Remove horizontal lines
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
        # no_lines = 255 - cv2.morphologyEx(255 - binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # contours, hierarchy = cv2.findContours(no_lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        ## if last digit of hierarchy, e.g. hierarchy[0][i][3] is -1 then it's parent, we should draw only if > threshold
        # no_dots = numpy.zeros_like(no_lines)
        ## Remove contours with area below threshold (removes random dots from the background)
        ## Gotta pay attention not to remove colon characters
        # filtered = []
        # for index, contour in enumerate(contours):
        #    area = cv2.contourArea(contour)
        #    # Always draw children, only draw parents if area large enough
        #    if hierarchy[0][index][3] != -1 or area > 40:
        #        filtered.append(contour)

        # cv2.drawContours(no_dots, filtered, -1, 255, -1)

        titles = ['Original', 'Red', 'Green', 'Binary Red', 'Binary Green', 'Merged']
        images = [rgb_img, img_onlyR, img_onlyG, r_binary, g_binary, merged]
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

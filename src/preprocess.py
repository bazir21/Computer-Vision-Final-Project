#!/usr/bin/env python3
import os
import cv2
import numpy
import argparse
from matplotlib import pyplot as plt


def preprocess(directory_name):
    images = os.listdir(directory_name)

    # Go through each image and process it with opencv
    for image_name in images:
        image_path = os.path.join(directory_name, image_name)
        img = cv2.imread(image_path)

        # Greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarisation
        #_, binary = cv2.threshold(grey, 220, 255, cv2.THRESH_BINARY_INV)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)

        # Remove horizontal lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
        no_lines = 255 - cv2.morphologyEx(255 - binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(no_lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # if last digit of hierarchy, e.g. hierarchy[0][i][3] is -1 then it's parent, we should draw only if > threshold
        no_dots = numpy.zeros_like(no_lines)
        # Remove contours with area below threshold (removes random dots from the background)
        # Gotta pay attention not to remove colon characters
        filtered = []
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # Always draw children, only draw parents if area large enough
            if hierarchy[0][index][3] != -1 or area > 40:
                filtered.append(contour)

        cv2.drawContours(no_dots, filtered, -1, 255, -1)

        titles = ['Original', 'Grey', 'Adaptive', 'No line', 'No dots']
        images = [img, grey, binary, no_lines, no_dots]
        for i in range(len(images)):
            plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
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

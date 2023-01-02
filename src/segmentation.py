
import os
import cv2 as cv
import numpy as np
import time

def prepare_training_data(dimensions = None):
    print("Preparing training data...")

    # y contains the ripeness score
    y = []

    # contains the images
    x = []

    # check if the resized images exist already
    resized_file_path = "../data/resized/images/" + str(dimensions[0]) + "x" + str(dimensions[1]) + "/"
    resized_ripeness_path = os.path.join("..", "data", "resized", "ripeness")
    os.makedirs(resized_ripeness_path, exist_ok=True)
    resized_ripeness_file = os.path.join(resized_ripeness_path, str(dimensions[0]) + "x" + str(dimensions[1]) + ".txt")

    start_time = time.time()

    if os.path.isdir(resized_file_path) and os.path.exists(resized_ripeness_file):
        print("Resized images found")
        # Make sure file list is sorted
        for filename in sorted(os.listdir(resized_file_path), key=lambda f: int(''.join(filter(str.isdigit, f)))):
            image_path = os.path.join(resized_file_path, filename)
            x.append(cv.imread(image_path))

        y = open(resized_ripeness_file, "r+")
        y = list(map(float, y.read().split(",")))

    else: # iterate through each file in the directory
        print("Generating new resized images...")
        directory = "../data/images/"
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            box_path = "../data/bounding_box/" + image_path[15:-4] + ".txt"
            ripeness, images = crop_with_box(cv.imread(image_path), box_path, dimensions)
            x += images
            y += ripeness

        # write new segmented images to resized images folder
        os.makedirs(resized_file_path, exist_ok=True)
        for count, resized_image in enumerate(x):
            cv.imwrite(resized_file_path + str(count + 1) + ".png", resized_image)

        # write ripeness levels to txt file
        with open(resized_ripeness_file, "w+") as f:
            f.write(", ".join([str(i) for i in y]))

    time_elapsed = time.time() - start_time
    print(f'Training data loaded, took {time_elapsed} seconds')

    # print(image_path[15:-4] + ",", end=" ")

    # TODO: Save the individual strawberries to avoid this extra processing. Take those strawberries directly.
    # IDK if it works or not it kinda does???!?!?!?
    # I swear to god i feel like the first initial run through is wrong and then the second loaded time is correct

    # view the strawberries :)
    # for image in x:
    #     cv.imshow("original", image)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()

    return x, y

def bounding_box_to_string_array(address):
    '''
    :param address: Address to bounding_box text fil
    :return: String array containing bounding_box details
    '''
    boxes = []

    with open(address) as f:
        [boxes.append(line) for line in f.readlines()]

    return boxes


def resize_strawberry(image, dimensions):
    # 227 being the required size of images for AlexNet
    # dimensions = (227, 227)
    image = cv.resize(image, dimensions, interpolation=cv.INTER_LINEAR)
    return image


def crop_with_box(image, address, dimensions):

    '''
    Crops each strawberry from image, given bounding box specifications.
    :param image: Original HD image
    :param address: Location of bounding_box text file
    :return: Array of cropped images containing strawberries to bouding_box details
    ## TODO: Implement the line below
    :return: Array of ripeness for each image
    '''

    # box - bounding_box text file
    # Bounding Box - each row contains:
    # [Ripeness Class ID(0 = unripe, 1 = partially ripe, 2 = fully ripe]
    # [Normalized x value for bounding box centre coordinate (left to right)]
    # [Normalized y value for bounding box centre coordinate (top to bottom)]
    # [Normalized x value for bounding box width]
    # [Normalized y value for bounding box width]

    # 0 0.65079 0.1164 0.03373 0.055556
    # 1 0.51984 0.11839 0.071429 0.12831
    # 0 0.46925 0.18519 0.061508 0.12169
    # 0 0.36756 0.14418 0.048611 0.10053
    boxes = bounding_box_to_string_array(address)

    cropped_images = []
    ripeness_scores = []
    for box in boxes:
        # why use list() ?:
        # https://stackoverflow.com/questions/6429638/how-to-split-a-string-of-space-separated-numbers-into-integers
        details = list(map(float, box.split()))
        if dimensions is not None:
            cropped_images.append(resize_strawberry(crop_image(image, details[1], details[2], details[3], details[4]),
                                                    dimensions))
        else:
            cropped_images.append(crop_image(image, details[1], details[2], details[3], details[4]))
        ripeness_scores.append(details[0])

    return ripeness_scores, cropped_images

    # 0 0.65079 0.1164 0.03373 0.055556
    # 756, 1008
    # 0 491, 117.3312

def crop_image(image, x, y, box_width, box_height):
    # img[80:280, 150:330]
    height, width = image.shape[0], image.shape[1]
    # [Normalized x value for bounding box centre coordinate (left to right)]
    # [Normalized y value for bounding box centre coordinate (top to bottom)]
    # [Normalized x value for bounding box width]
    # [Normalized y value for bounding box width]
    box_width *= width
    box_height *= height
    x = (x * width) - (box_width / 2)
    y = (y * height) - (box_height / 2)
    x2 = x + box_width
    y2 = y + box_height

    return image[round(y):round(y2), round(x):round(x2)]

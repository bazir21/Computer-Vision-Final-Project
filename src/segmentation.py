
import os
import cv2 as cv

def prepare_training_data():
    # y contains the ripeness score
    y = []

    # contains the images
    x = []

    # iterate through each file in the directory
    directory = "../data/images/"
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        box_path = "../data/bounding_box/" + image_path[15:-4] + ".txt"
        ripeness, images = crop_with_box(cv.imread(image_path), box_path)
        x = x + images
        y = y + ripeness
        print(image_path[15:-4] + ",", end=" ")

    # TODO: Save the individual strawberries to avoid this extra processing. Take those strawberries directly.

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


def crop_with_box(image, address):

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

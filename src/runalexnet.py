from models.alexnet import alexnet
from segmentation import prepare_training_data

x, y = prepare_training_data(dimensions=(227, 227))
alexnet(x, y, True)

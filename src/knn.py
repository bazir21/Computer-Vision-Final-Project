import numpy as np
from segmentation import prepare_training_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def knn():
    x, y = prepare_training_data(dimensions=(32, 32))
    x = np.array(x)
    x = x.reshape((x.shape[0], (x.shape[1] * x.shape[2] * x.shape[3])))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test)))

knn()

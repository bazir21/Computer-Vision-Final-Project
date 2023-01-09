import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

from segmentation import prepare_training_data

NUM_SPLITS = 5


def main():
    # x is list of cropped images, y is numeric label corresponding to ripeness level
    x, y = prepare_training_data(dimensions=(32, 32))
    x = np.array(x)
    # Flatten each image, so we have w * h * c number of features
    x = x.reshape((x.shape[0], (x.shape[1] * x.shape[2] * x.shape[3])))

    y = np.array(y, dtype=np.uint8)

    kf = KFold(n_splits=NUM_SPLITS)
    print(f"Running {NUM_SPLITS}-fold cross-validation...")
    Cs = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    for C in Cs:
        print(f"Running model with C={C}")
        model = LinearSVC(C=C, max_iter=10000)
        accuracy = []

        start_time = time.time()
        for train, test in kf.split(y):
            model.fit(x[train], y[train])
            y_pred = model.predict(x[test])
            accuracy.append(accuracy_score(y[test], y_pred))

        taken = time.time() - start_time
        print(f"Took {taken}s")

        mean_accuracy = np.array(accuracy).mean()
        std = np.array(accuracy).std()

        print(f'Mean accuracy with C={C}: {mean_accuracy}')
        print(f'STD: {std}')

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # model = SVC(kernel="linear")
    # model = KNeighborsClassifier(n_neighbors=7)
    # model.fit(x_train, y_train)
    # print(classification_report(y_test, model.predict(x_test)))


if __name__ == "__main__":
    main()

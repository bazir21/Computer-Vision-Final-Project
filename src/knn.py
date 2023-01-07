import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from segmentation import prepare_training_data
from conf_matrix import confusion_matrix_graph


# Inspired by code from here:
# https://pyimagesearch.com/2021/04/17/your-first-image-classifier-using-k-nn-to-classify-images/

def main():
    x, y = prepare_training_data(dimensions=(32, 32))
    x = np.array(x)
    x = x.reshape((x.shape[0], (x.shape[1] * x.shape[2] * x.shape[3])))

    y = np.array(y, dtype=np.uint8)

    kf = KFold(n_splits=5)

    # Uncomment following loops to perform hyperparameter selection via cross-validation
    # for k in range(5, 15):
    # for metric in ["euclidean", "cosine"]:
    # print(f"Running with {metric} metric and k={k}")

    k = 7
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    # Store through each fold to calculate confusion matrix and accuracy
    targets = []
    predictions = []
    accuracy = []

    start_time = time.time()
    for train, test in kf.split(y):
        model.fit(x[train], y[train])
        y_pred = model.predict(x[test])
        accuracy.append(accuracy_score(y[test], y_pred))
        targets.extend(y[test])
        predictions.extend(y_pred)
        print("Completed a fold...")

    mean_value = np.array(accuracy).mean()
    std = np.array(accuracy).std()
    taken = time.time() - start_time

    print(f'Took {taken}s')
    print(f'Mean: {mean_value}')
    print(f'STD: {std}')

    # YOLOv5 normalises over the columns, where the columns are the true values and rows predicted
    # For YOLO it's predicted over true, and normalised by column (i.e. true labels)
    conf_matrix = confusion_matrix(targets, predictions, normalize="true").T  # Transpose to be same format as YOLO
    print(np.array_str(conf_matrix, precision=3))
    # Save confusion matrix graph
    confusion_matrix_graph(conf_matrix, ["Unripe", "Partially Ripe", "Ripe"], "knn_confusion_matrix.png")


if __name__ == "__main__":
    main()

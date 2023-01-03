import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from segmentation import prepare_training_data


# all ripped straight from here lol! 🥴🥴
# https://pyimagesearch.com/2021/04/17/your-first-image-classifier-using-k-nn-to-classify-images/

def main():
    x, y = prepare_training_data(dimensions=(32, 32))
    x = np.array(x)
    x = x.reshape((x.shape[0], (x.shape[1] * x.shape[2] * x.shape[3])))

    y = np.array(y, dtype=np.uint8)

    kf = KFold(n_splits=5)
    for k in range(1, 20):
        print(f"Running with k={k}")
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

        accuracy = []

        start_time = time.time()
        for train, test in kf.split(y):
            model.fit(x[train], y[train])
            y_pred = model.predict(x[test])
            accuracy.append(accuracy_score(y[test], y_pred))

        mean_value = np.array(accuracy).mean()
        std = np.array(accuracy).std()
        taken = time.time() - start_time

        print(f'Took {taken}s')
        print(f'Mean: {mean_value}')
        print(f'STD: {std}')


if __name__ == "__main__":
    main()

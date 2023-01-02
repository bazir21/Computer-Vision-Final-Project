import numpy as np
from segmentation import prepare_training_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

# all ripped straight from here lol! 🥴🥴
# https://pyimagesearch.com/2021/04/17/your-first-image-classifier-using-k-nn-to-classify-images/

def knn():
    x, y = prepare_training_data(dimensions=(32, 32))
    x = np.array(x)
    x = x.reshape((x.shape[0], (x.shape[1] * x.shape[2] * x.shape[3])))

    y = np.array(y, dtype=np.uint8)

    kf = KFold(n_splits=5)
    # model = SVC(kernel="linear")
    model = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)

    accuracy = []

    for train, test in kf.split(y):
        model.fit(x[train], y[train])
        y_pred = model.predict(x[test])
        accuracy.append(accuracy_score(y[test], y_pred))

    mean_value = np.array(accuracy).mean()
    std = np.array(accuracy).std()

    print(f'Mean: {mean_value}')
    print(f'STD: {std}')

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # model = SVC(kernel="linear")
    # model = KNeighborsClassifier(n_neighbors=7)
    # model.fit(x_train, y_train)
    # print(classification_report(y_test, model.predict(x_test)))

knn()

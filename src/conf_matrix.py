import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# Code for making a confusion matrix graph in the same style as YOLOv5's
# https://github.com/ultralytics/yolov5/blob/fdc35b119ad21c7f205596dbb238f780c87040ec/utils/metrics.py#L187

def confusion_matrix_graph(matrix, labels, save_path="confusion_matrix.png"):
    matrix[matrix < 0.005] = np.nan
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc = 3
    nn = len(labels)

    sb.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    names = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
    ticklabels = labels if names else "auto"
    sb.heatmap(matrix,
               ax=ax,
               annot=nc < 30,
               annot_kws={
                   "size": 20},
               cmap='Blues',
               fmt='.2f',
               square=True,
               vmin=0.0,
               xticklabels=ticklabels,
               yticklabels=ticklabels).set_facecolor((1, 1, 1))
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    fig.savefig(save_path, dpi=250)


def main():
    matrix = np.array([
        [0.9959, 0.1741, 0.05736],
        [0.003042, 0.7663, 0.2431],
        [0.00008, 0.05953, 0.6995]
    ])
    confusion_matrix_graph(matrix, ["Unripe", "Partially Ripe", "Ripe"], "knn_cf.png")


if __name__ == "__main__":
    main()

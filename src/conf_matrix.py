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
                   "size": 12},
               cmap='Blues',
               fmt='.2f',
               square=True,
               vmin=0.0,
               xticklabels=ticklabels,
               yticklabels=ticklabels).set_facecolor((1, 1, 1))
    ax.set_ylabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    fig.savefig(save_path, dpi=250)
    plt.show()

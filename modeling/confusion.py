import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix



OUT_BASE = "out/conf"

def save_confusion_m(y_pred, true_y, out_name, accuracy, k = 10):
    if not os.path.exists(OUT_BASE):
        os.makedirs(OUT_BASE)
    confusion_m = confusion_matrix(true_y, y_pred)
    y_labels = ["{}-{}".format(i * 5 / k, (i + 1) * 5 / k) for i in np.unique(true_y)]
    plt.figure()
    ax = sb.heatmap(pd.DataFrame(confusion_m, columns=y_labels,
                                 index=y_labels), annot=True, cbar=False, fmt="d")
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)
    plt.gcf().subplots_adjust(bottom=0.25, left=0.20)
    plt.title("{}, accuracy = {:.3f}".format(out_name, accuracy))
    plt.savefig("{}/{}.png".format(OUT_BASE, out_name))
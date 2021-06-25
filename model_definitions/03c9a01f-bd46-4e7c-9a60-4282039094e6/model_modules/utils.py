import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.special import expit
invlogit=expit

def wrap(arr):
    return np.ascontiguousarray(arr)

def _decode_data(X, feature_names, category_map):
    """
    Given an encoded data matrix `X` returns a matrix where the
    categorical levels have been replaced by human readable categories.
    """

    X_new = np.zeros(X.shape, dtype=object)
    for idx, name in enumerate(feature_names):
        categories = category_map.get(idx, None)
        if categories:
            for j, category in enumerate(categories):
                encoded_vals = X[:, idx] == j
                X_new[encoded_vals, idx] = category
        else:
            X_new[:, idx] = X[:, idx]

    return X_new

def plot_conf_matrix(y_test, y_pred, class_names):
    """
    Plots confusion matrix. Taken from:
    http://queirozf.com/entries/visualizing-machine-learning-models-examples-with-scikit-learn-and-matplotlib
    """

    matrix = confusion_matrix(y_test,y_pred)


    # place labels at the top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')

    # plot the matrix per se
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

    # plot colorbar to the right
    plt.colorbar()

    fmt = 'd'

    # write the number of predictions in each bucket
    thresh = matrix.max() / 2.
    for i, j in product(range(matrix.shape[0]), range(matrix.shape[1])):

        # if background is dark, use a white number, and vice-versa
        plt.text(j, i, format(matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if matrix[i, j] > thresh else "black")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label',size=14)
    plt.xlabel('Predicted label',size=14)
    plt.show()

def predict(xgb_model, dataset, proba=False, threshold=0.5):
    """
    Predicts labels given a xgboost model that outputs raw logits.
    """

    y_pred = model.predict(dataset)  # raw logits are predicted
    y_pred_proba = invlogit(y_pred)
    if proba:
        return y_pred_proba
    y_pred_class = np.zeros_like(y_pred)
    y_pred_class[y_pred_proba >= threshold] = 1  # assign a label

    return y_pred_class

def _get_importance(model, measure='weight'):
    """
    Retrieves the feature importances from an xgboost
    models, measured according to the criterion `measure`.
    """

    imps = model.get_score(importance_type=measure)
    names, vals = list(imps.keys()), list(imps.values())
    sorter = np.argsort(vals)
    s_names, s_vals = tuple(zip(*[(names[i], vals[i]) for i in sorter]))

    return s_vals[::-1], s_names[::-1]

def plot_importance(feat_imp, feat_names, ax=None, **kwargs):
    """
    Create a horizontal barchart of feature effects, sorted by their magnitude.
    """

    left_x, step ,right_x = kwargs.get("left_x", 0), kwargs.get("step", 50), kwargs.get("right_x")
    xticks = np.arange(left_x, right_x, step)
    xlabel = kwargs.get("xlabel", 'Feature effects')
    xposfactor = kwargs.get("xposfactor", 1)
    textfont = kwargs.get("text_fontsize", 25) # 16
    yticks_fontsize = kwargs.get("yticks_fontsize", 25)
    xlabel_fontsize = kwargs.get("xlabel_fontsize", 30)
    textxpos = kwargs.get("textxpos", 60)
    textcolor = kwargs.get("textcolor", 'white')

    if ax:
        fig = None
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(feat_imp))
    ax.barh(y_pos, feat_imp)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=yticks_fontsize)
    ax.set_xticklabels(xticks, fontsize=30, rotation=45)
    ax.invert_yaxis()                  # labels read top-to-bottom
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_xlim(left=left_x, right=right_x)

    for i, v in enumerate(feat_imp):
#         if v<0:
        textxpos = xposfactor*textxpos
        ax.text(v - textxpos, i + .25, str(round(v, 3)), fontsize=textfont, color=textcolor)
    return ax, fig

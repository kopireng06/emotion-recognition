import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend
import tensorflow as tf

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    if cmap is None:
        cmap = plt.get_cmap('Oranges')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylim(len(target_names)-0.5, -0.5)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig(title + '.png', dpi=500, bbox_inches = 'tight')
    plt.show()


def evaluate_model(model, dataset, class_labels):
    y_true = []
    y_pred = []
    for x,y in dataset:
      y_true.append(y)
      y_pred.append(tf.argmax(model.predict(x),axis = 1))
      
    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.concat(y_true, axis=0)

    cm_train = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=class_labels))
    plot_confusion_matrix(cm_train, class_labels)


def specificity(y_true, y_pred):    
    true_negatives = backend.sum(backend.round(backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = backend.sum(backend.round(backend.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + backend.epsilon())
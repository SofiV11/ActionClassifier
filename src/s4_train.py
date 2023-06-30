'''
1. Загрузка характеристик и меток из файлов csv
2. Обучение модели
3. Сохранение модели в папке `model/`.
'''

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import classification_report

if True:
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_classifier import ClassifierOfflineTrain



def par(path):
    return ROOT + path if (path and path[0] != "/") else path

cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s4_train.py"]

CLASSES = np.array(cfg_all["classes"])


SRC_PROCESSED_FEATURES = par(cfg["input"]["processed_features"])
SRC_PROCESSED_FEATURES_LABELS = par(cfg["input"]["processed_features_labels"])

DST_MODEL_PATH = par(cfg["output"]["model_path"])

def train_test_split(X, Y, ratio_of_test_size):
    ''' Разделить учебные данные по соотношению '''
    IS_SPLIT_BY_SKLEARN_FUNC = True

    if IS_SPLIT_BY_SKLEARN_FUNC:
        RAND_SEED = 1
        tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
            X, Y, test_size=ratio_of_test_size, random_state=RAND_SEED)

    else: # Сделать тренировку/тест одинаковыми.
        tr_X = np.copy(X)
        tr_Y = Y.copy()
        te_X = np.copy(X)
        te_Y = Y.copy()
    return tr_X, te_X, tr_Y, te_Y

def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):
    ''' Оценить точность и временные затраты '''

    # Точность
    t0 = time.time()

    print(f'Model name: {model.model_name}')

    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}") # рассчитывается как доля правильных предсказаний

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")

    print("Accuracy report:")
    print(classification_report(
        te_Y, te_Y_predict, target_names=classes, output_dict=False))

    # Временные затраты
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Time cost for predicting one sample: "
          "{:.5f} seconds".format(average_time))

    # Матрица ошибок
    axis, cf = lib_plot.plot_confusion_matrix(
        te_Y, te_Y_predict, classes, normalize=False, size=(12, 8))
    plt.show()



def main():

    # Загрузка предварительно обработанных данных
    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(SRC_PROCESSED_FEATURES, dtype=float)  # features
    Y = np.loadtxt(SRC_PROCESSED_FEATURES_LABELS, dtype=int)  # labels
    
    # Разделение на обучающую и тестовую выборку
    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, ratio_of_test_size=0.3)
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    # Обучение
    print("\nStart training model ...")
    model = ClassifierOfflineTrain(model_name='Nearest Neighbors')
    model.train(tr_X, tr_Y)

    # Оценка
    print("\nStart evaluating model ...")
    evaluate_model(model, CLASSES, tr_X, tr_Y, te_X, te_Y)

    # Сохранение
    print("\nSave model to " + DST_MODEL_PATH.format(model.model_name))
    with open(DST_MODEL_PATH.format(model.model_name), 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()

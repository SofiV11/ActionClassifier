'''
Загрузка данных о скелете из файла `skeletons_info.txt`,
Обработка данных,
Сохранение характеристик и метк в файл .csv.
'''

import numpy as np

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_commons as lib_commons
    from utils.lib_skeletons_io import load_skeleton_data
    from utils.lib_feature_proc import extract_multi_frame_features


def par(path):
    return ROOT + path if (path and path[0] != "/") else path

cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s3_preprocess_features.py"]

CLASSES = np.array(cfg_all["classes"])

# Распознавание действий
WINDOW_SIZE = int(cfg_all["features"]["window_size"]) # количество кадров, используемых для извлечения признаков

# Input, output
SRC_ALL_SKELETONS_TXT = par(cfg["input"]["all_skeletons_txt"])
DST_PROCESSED_FEATURES = par(cfg["output"]["processed_features"])
DST_PROCESSED_FEATURES_LABELS = par(cfg["output"]["processed_features_labels"])



def process_features(X0, Y0, video_indices, classes):
    ''' Обработка характеристик '''
    #  Преобразование признаков
    # From: необработанные признаки отдельного изображения.
    # To:   временные характеристики, рассчитанные из нескольких необработанных характеристик,
    #       нескольких соседних изображений, включая скорость, нормализованную позицию и т.д.
    ADD_NOISE = False
    if ADD_NOISE:
        X1, Y1 = extract_multi_frame_features(
            X0, Y0, video_indices, WINDOW_SIZE, 
            is_adding_noise=True, is_print=True)
        X2, Y2 = extract_multi_frame_features(
            X0, Y0, video_indices, WINDOW_SIZE,
            is_adding_noise=False, is_print=True)
        X = np.vstack((X1, X2))
        Y = np.concatenate((Y1, Y2))
        return X, Y
    else:
        print('def_x')
        print (X0)
        X, Y = extract_multi_frame_features(
            X0, Y0, video_indices, WINDOW_SIZE, 
            is_adding_noise=False, is_print=True)

        print('new_x')
        print(X)
        return X, Y

# -- Main


def main():
    # Загрузка данных
    X0, Y0, video_indices = load_skeleton_data(SRC_ALL_SKELETONS_TXT, CLASSES)

    # Обработка характеристик
    print("\nExtracting time-serials features ...")
    X, Y = process_features(X0, Y0, video_indices, CLASSES)
    print(f"X.shape = {X.shape}, len(Y) = {len(Y)}")

    # Сохранение
    print("\nWriting features and labels ...")

    os.makedirs(os.path.dirname(DST_PROCESSED_FEATURES), exist_ok=True)
    os.makedirs(os.path.dirname(DST_PROCESSED_FEATURES_LABELS), exist_ok=True)

    np.savetxt(DST_PROCESSED_FEATURES, X, fmt="%.5f")
    print("Save features to: " + DST_PROCESSED_FEATURES)

    np.savetxt(DST_PROCESSED_FEATURES_LABELS, Y, fmt="%i")
    print("Save labels to: " + DST_PROCESSED_FEATURES_LABELS)


if __name__ == "__main__":
    main()

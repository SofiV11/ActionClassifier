'''
Считывает несколько txt скелетов и сохраняет их в один txt.
Если изображение не имеет скелета, оно отбрасывается.
Если метка изображения не принадлежит `CLASSES`, наблюдение отбрасывается.

Input:
    `skeletons/1.txt` ~ `skeletons/xxxx.txt` from `SRC_DETECTED_SKELETONS_FOLDER`.
Output:
    `skeletons_info.txt`. The filepath is `DST_ALL_SKELETONS_TXT`.
'''

import numpy as np
import simplejson
import collections

if True:
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))[:-3]
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"\\"
    sys.path.append(ROOT)

    import utils.lib_commons as lib_commons


def par(path):  # Добавляет ROOT к пути, если он не абсолютный
    return ROOT + path if (path and path[0] != "/") else path

# Конфигурации
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s2_put_skeleton_txts_to_a_single_txt.py"]

CLASSES = np.array(cfg_all["classes"])

SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

SRC_DETECTED_SKELETONS_FOLDER = par(cfg["input"]["detected_skeletons_folder"])
DST_ALL_SKELETONS_TXT = par(cfg["output"]["all_skeletons_txt"])

IDX_PERSON = 0  # Only use the skeleton of the 0th person in each image
IDX_ACTION_LABEL = 3  # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]

# чтение каждого скелета
def read_skeletons_from_ith_txt(i):
    ''' 
    Аргументы:
         i {int}: i-й скелетный txt. Индекс с нуля.
    Возвращает:
        skeletons_in_ith_txt {list of list}:
           Длина данных каждого скелета должна быть 41 = 5 изображений + 36 xy позиций
    '''
    filename = SRC_DETECTED_SKELETONS_FOLDER + \
        SKELETON_FILENAME_FORMAT.format(i)
    skeletons_in_ith_txt = lib_commons.read_listlist(filename)
    return skeletons_in_ith_txt


def get_length_of_one_skeleton_data(filepaths):
    ''' Нахождение непустого txt-файла, а затем получение длины одного скелета данных
     Длина данных должна быть равна 41, где:
    41 = 5 + 36.
        5: [cnt_action, cnt_clip, cnt_image, action_label, filepath].
        36: 18 суставов * 2 позиции xy
    '''
    for i in range(len(filepaths)):
        skeletons = read_skeletons_from_ith_txt(i)
        if len(skeletons):
            skeleton = skeletons[IDX_PERSON]
            data_size = len(skeleton)
            assert(data_size == 41)
            return data_size
    raise RuntimeError(f"No valid txt under: {SRC_DETECTED_SKELETONS_FOLDER}.")


# -- Main
if __name__ == "__main__":
    ''' Чтение нескольких txt скелетов и сохранение их в один txt.'''

    # Получение имен файлов скелета
    filepaths = lib_commons.get_filenames(SRC_DETECTED_SKELETONS_FOLDER,
                                          use_sort=True, with_folder_path=True) # получение списка абсолютных путей ко всем файлам в директории
    num_skeletons = len(filepaths)

    # Проверка длины данных одного скелета
    data_length = get_length_of_one_skeleton_data(filepaths)
    print("Data length of one skeleton is {data_length}")

    # переход к all_skeletons
    all_skeletons = []
    labels_cnt = collections.defaultdict(int)
    for i in range(num_skeletons):

        skeletons = read_skeletons_from_ith_txt(i) # возвращает элемент списка с данными по 1 скелету
        if not skeletons:  # Если пусто, удаляем
            continue
        skeleton = skeletons[IDX_PERSON] #IDX_PERSON - индекс позиции кол-ва скелетов = [0]
        label = skeleton[IDX_ACTION_LABEL]
        if label not in CLASSES:  # Если класс недействителен, удаляем
            continue
        labels_cnt[label] += 1

        # Добавление
        all_skeletons.append(skeleton)

        # Лог
        if i == 1 or i % 100 == 0:
            print("{}/{}".format(i, num_skeletons))

    # Сохранение
    with open(DST_ALL_SKELETONS_TXT, 'w') as f: # режим записи, записываем в файл json данные
        simplejson.dump(all_skeletons, f)

    print(f"There are {len(all_skeletons)} skeleton data.")
    print(f"They are saved to {DST_ALL_SKELETONS_TXT}")
    print("Number of each action: ")
    for label in CLASSES:
        print(f"    {label}: {labels_cnt[label]}")

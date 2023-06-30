'''
Подпрограмма 1
Считывание обучающих изображений на основе `valid_images.txt`, определение скелетов и сохранение результатов.

На каждом изображении должен быть только 1 человек, выполняющий одно действие.
Каждое изображение называется 1.jpg, 2.jpg, ...

Input:
    SRC_IMAGES_DESCRIPTION_TXT
    SRC_IMAGES_FOLDER
    
Output:
    DST_IMAGES_INFO_TXT
    DST_DETECTED_SKELETONS_FOLDER
    DST_VIZ_IMGS_FOLDER
'''

import cv2

# инициализация корневого пути и текущего
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))[:-3]
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+'\\'
sys.path.append(ROOT)

# Импорт утилит
from utils.lib_openpose import SkeletonDetector
from utils.lib_tracker import Tracker
from utils.lib_skeletons_io import ReadValidImagesAndActionTypesByTxt
import utils.lib_commons as lib_commons


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path


# Конфигурации


cfg_all = lib_commons.read_yaml(ROOT + "config\\config.yaml")
cfg = cfg_all["s1_get_skeletons_from_training_imgs.py"]

IMG_FILENAME_FORMAT = cfg_all["image_filename_format"]
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

# Пути входных данных
if True:
    SRC_IMAGES_DESCRIPTION_TXT = par(cfg["input"]["images_description_txt"])
    SRC_IMAGES_FOLDER = par(cfg["input"]["images_folder"])

# Пути к результирующим данным
if True:
    # В этом txt будет храниться информация об изображении, такая как индекс, метка действия, имя файла и т.д.
    DST_IMAGES_INFO_TXT = par(cfg["output"]["images_info_txt"])

    # В каждом txt будет храниться скелет каждого изображения
    DST_DETECTED_SKELETONS_FOLDER = par(
        cfg["output"]["detected_skeletons_folder"])

    # Визуализация. Каждое изображение отрисовывается с помощью обнаруженного скелета
    DST_VIZ_IMGS_FOLDER = par(cfg["output"]["viz_imgs_folders"])

# Openpose
if True:
    OPENPOSE_MODEL = cfg["openpose"]["model"]
    OPENPOSE_IMG_SIZE = cfg["openpose"]["img_size"]

import tensorflow as tf


class ImageDisplayer(object):
    def __init__(self):
        self._window_name = "cv2_display_window"
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

    def display(self, image, wait_key_ms=0):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):

        cv2.destroyWindow(self._window_name)


# -- Main
if __name__ == "__main__":

    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=conf)
    tf.compat.v1.keras.backend.set_session(session)

    # Инициаилизация детектора из lib_openpose
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    # Инициализация трекера из lib_tracker
    multiperson_tracker = Tracker()

    # Инициализация объекта чтения изображений
    images_loader = ReadValidImagesAndActionTypesByTxt(
        img_folder=SRC_IMAGES_FOLDER,
        valid_imgs_txt=SRC_IMAGES_DESCRIPTION_TXT,
        img_filename_format=IMG_FILENAME_FORMAT
    )

    #
    os.makedirs(os.path.dirname(DST_IMAGES_INFO_TXT), exist_ok=True)
    os.makedirs(DST_DETECTED_SKELETONS_FOLDER, exist_ok=True)
    os.makedirs(DST_VIZ_IMGS_FOLDER, exist_ok=True)

    # -- Считывание изображений и их обработка
    num_total_images = images_loader.num_images
    # ГЛАВНЫЙ ЦИКЛ ПО КАДРАМ-ИЗОБРАЖЕНИЯМ

    with tf.device('/GPU:0'):
        for ith_img in range(num_total_images):
            img, str_action_label, img_info = images_loader.read_image()  # Возвращает нампи массив изображения, лейбл действия, и элемент массива инфо
            # Детектирование
            humans = skeleton_detector.detect(img)  # Детектирование людей на изображении, возвращает класс

            # Отрисовка
            img_disp = img.copy()
            skeleton_detector.draw(img_disp, humans)
            # img_displayer.display(img_disp, wait_key_ms=1)

            # Получение данных о скелете и сохранение в файл
            skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)  # принимает класс-человека,
            # возвращает скелет - список из 36 элементов (18 суставов * 2 значения координат)
            # и результирующую высоту
            dict_id2skeleton = multiperson_tracker.track(
                skeletons)  # dict: (int human id) -> (np.array() skeleton)
            skels_to_save = [img_info + skeleton.tolist()
                             for skeleton in dict_id2skeleton.values()]

            # Сохранение результатов

            # Сохранение скелетных данных для обучения
            filename = SKELETON_FILENAME_FORMAT.format(ith_img)
            lib_commons.save_listlist(
                DST_DETECTED_SKELETONS_FOLDER + filename,
                skels_to_save)

            # Сохранение визуализированного изображения для отладки
            filename = IMG_FILENAME_FORMAT.format(ith_img)
            cv2.imwrite(
                DST_VIZ_IMGS_FOLDER + filename,
                img_disp)

            print(f"{ith_img}/{num_total_images} th image "
                  f"has {len(skeletons)} people in it")

    print("Program 1 ends")

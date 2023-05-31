import cv2

if True:  # Include project path
    import sys
    import os

    ROOT = os.path.dirname(os.path.abspath(__file__))
    CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
    sys.path.append(ROOT)

    # from utils.lib_openpose import SkeletonDetector
    # from utils.lib_tracker import Tracker
    # from utils.lib_skeletons_io import ReadValidImagesAndActionTypesByTxt
    import utils.lib_commons as lib_commons



def reduce_fps(path):
    # Настройка файла
    # dir = ROOT[:-7] + 'data\\data_in'
    dir = r'E:\pythonProject\Realtime-Action-Recognition\data\video_data_in'

    for filename in os.listdir(dir):
        # videoFile = dir +  filename# r"D:\274.avi"  # Путь к файлу
        vide_file = os.path.join(dir, filename)
        vidcap = cv2.VideoCapture(vide_file)
        success, image = vidcap.read()

        seconds = 0.1  # время
        fps = vidcap.get(cv2.CAP_PROP_FPS)  # Получаем кадры в секунду
        multiplier = fps * seconds

        while success:
            frameId = int(round(vidcap.get(1)))  # текущий номер кадра, округленный
            success, image = vidcap.read()

        if frameId % multiplier == 0:
            cv2.imwrite_(r"E:\pythonProject\Realtime-Action-Recognition\data\data_out\frame%d.jpg" % frameId, image)

        vidcap.release()
        print("Завершено")
        frames = r'E:\pythonProject\Realtime-Action-Recognition\data\data_out'
        frames = [os.path.join(frames, 'frame{}.jpg'.format(i)) for i in
                  range(5, 3750)]  # Путь к скриншотам. Нужно указать начальный номер кадра и конечный.
        frame = cv2.imread(frames[0])
        writer = cv2.VideoWriter(
            filename + '.mp4',
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),  # кодек
            10.0,  # fps
            (960, 600),  # ширина, высота кадра
            isColor=1)

        for frame in map(cv2.imread, frames):
            writer.write(frame)
        writer.release()
        cv2.destroyAllWindows()


def reduce_fps2(filename):
    # Открываем видеофайл
    # dir = r'E:\pythonProject\Realtime-Action-Recognition\data\video_data_in' + filename
    video_path = r'E:\pythonProject\Realtime-Action-Recognition\data\video_data_in' + filename
    video = cv2.VideoCapture(video_path)
    # Создаем новый видеофайл с измененным числом кадров
    output_path = r'E:\pythonProject\Realtime-Action-Recognition\data\data_out\ss.mp4'
    output_fps = 30  # Новое число кадров в секунду
    # Получаем текущее число кадров в секунду в исходном видео
    fps = video.get(cv2.CAP_PROP_FPS)
    # Получаем размер кадра в исходном видео
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # Создаем объект для записи видео с новым числом кадров
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, frame_size)
    # Обрабатываем каждый кадр в исходном видео
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Здесь вы можете выполнить любую обработку кадра, если необходимо

        # Записываем обработанный кадр в новый видеофайл
        output_video.write(frame)
    # Закрываем видеофайлы
    video.release()
    output_video.release()


import tools.video2images as video2images
def merge_data():
    # Настройка файла
    import tools.video2images as video2images
    input_dir = ROOT + '\\data_in\\'
    output_dir = ROOT + '\\data_out\\' # можно добавить в конфиг
    existing_data_dir = ROOT + '\\source_images3\\' # можно взять из конфига

    # Как мерджить данные, обзор содержимого папки с существующими данными, вырезать типы активности до знака "_", и дальше кидать аналогичное туда,
    # а если класс действия из видео не подходит ни для одного действия из списка существующих, то создать новую папку
    # (?) создать dict c действием и действием из внешних добавлемых данных
    # (?) есть туева хуча видео, нужно по циклу пройтись по ним для анализа наличия действий, что можно закинуть в существующее место, новинки так называемые, \
    # вычислить, создать для них папки и кидать туда кадры а как я уже запуталась пойду спать 
    # (?) есть ли тайная задумка в названиях папок с файлами
    #


    for filename in os.listdir(input_dir):
        # videoFile = dir +  filename# r"D:\274.avi"  # Путь к файлу
        # video_file = os.path.join(input_dir, filename)
        start_p = filename.find('_', filename.find('_')+1)
        end_p = filename.find('_', start_p+1)
        act_name = filename[start_p+1:end_p]
        if act_name == 'wave1':
            act_name = 'hello'

        if act_name == 'wave2':
            act_name = 'wave'

        if act_name == 'sit-down':
            act_name = 'sit'

        if act_name == 'standing':
            act_name = 'stand'

        if act_name == 'box':
            act_name = 'punch'

        if act_name == 'pjump':
            act_name = 'jump'

        cnf = lib_commons.read_yaml(ROOT[:-5] + "\\config\\config.yaml")
        classes = cnf['classes']
        # try:
        #     # classes.index("" + act_name + "")
        #     for e_folder in os.listdir(existing_data_dir):
        #         if e_folder.find("" + act_name + "") >= 0:
        #             subdir_name = (existing_data_dir + e_folder)
        #             break
        # except:
        #     subdir_name = existing_data_dir + act_name
        #     # if not subdir_name:
        #     #   os.mkdir(existing_data_dir + subdir_name)
        #     if not os.path.exists(existing_data_dir + subdir_name):
        #         os.makedirs(existing_data_dir + subdir_name)

        subdir_name = ''
        for e_folder in os.listdir(existing_data_dir):
            if e_folder.find("" + act_name + "") >= 0:
                subdir_name = (existing_data_dir + e_folder)
                break
        if not subdir_name:
            subdir_name = existing_data_dir + act_name
            # if not subdir_name:
            #   os.mkdir(existing_data_dir + subdir_name)
            if not os.path.exists(subdir_name):
                os.makedirs(subdir_name)

        args = video2images.parse_args()
        args.input_video_path = input_dir + filename
        args.output_folder_path = subdir_name + '\\'
        video2images.main(args)


# пофиксить момент с существующими папками ------- ----------------------вроде ок
# пофиксить момент с интервалом в зависимости от продолжительности видео ------хз но вроде ок
# пофиксить момент с валидным текстом в зависимости от интервала и продолжительности определить количество кадров и сложить их для одного типа движения или сделать вручную
#
#

def rename_file1(dir, start=0):
    cfg_all = lib_commons.read_yaml(ROOT[:-4] + "config\\config.yaml")
    image_format = cfg_all["image_filename_format"]
    files = []
    files = os.listdir(dir)
    files.sort()
    if start != 0:
        start = int(max(os.listdir(dir)).split('.')[0])
    for i in files:
        os.rename(dir + '/' + i,
                  (dir + '/' + image_format.format(int(i.split('.')[0])+100000)))
    files.sort()
    j=0
    for i in files:
        j=j+1
        os.rename(dir + '/' + i,
                  (dir + '/' + image_format.format(j)))
        files.sort()


def rename_file(dir, start=0):
    cfg_all = lib_commons.read_yaml(ROOT[:-4] + "config\\config.yaml")
    image_format = cfg_all["image_filename_format"]
    files = []
    files = os.listdir(dir)
    files.sort()
    if start != 0:
        start = int(max(os.listdir(dir)).split('.')[0])
    for i in files:
        os.rename(dir + '/' + i,
                  (dir + '/' + image_format.format(int(i.split('.')[0]))))
        files.sort()


if __name__ == "__main__":

     path = 'E:/pythonProject/Realtime-Action-Recognition/data/source_images3/'
     pathlist = os.listdir(path)
     for direct in pathlist:
         rename_file(path+direct, 0)

     merge_data()

     path = 'E:/pythonProject/Realtime-Action-Recognition/data/source_images3/'
     pathlist = os.listdir(path)
     for direct in pathlist:
         rename_file1(path+direct, 0)


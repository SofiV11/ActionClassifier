import cv2
import yaml

if True:  # Include project path
    import sys
    import os

    ROOT = os.path.dirname(os.path.abspath(__file__))
    CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
    sys.path.append(ROOT)

    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_skeletons_io import ReadValidImagesAndActionTypesByTxt
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




if __name__ == "__main__":
     # reduce_fps(10)
     reduce_fps2('isldas_federico_hands-clap_181.avi')

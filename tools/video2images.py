'''
Description:
    Convert a video to a folder of images.
    
Example of usage:
    python video2images.py \
        -i /home/feiyu/Desktop/learn_coding/test_data/video_of_waving_object.avi \
        -o /home/feiyu/Desktop/learn_coding/test_data/video_convert_result \
        --sample_interval 2 \
        --max_frames 30
'''

import numpy as np
import cv2
import sys
import os
import time
import numpy as np
import simplejson
import sys
import os
import csv
import glob
import argparse
import itertools
import utils.lib_commons as lib_commons
import imutils


ROOT = os.path.dirname(os.path.abspath(__file__))[:-5]
VALID_ING_TXT = os.path.dirname(os.path.abspath(__file__))[:-5] + "\\data\\source_images3\\valid_images.txt"
cfg_all = lib_commons.read_yaml(ROOT + "config\\config.yaml")
cfg_img_size = cfg_all['img_size'].split('x')
img_h = int(cfg_img_size[0])
img_w = int(cfg_img_size[1])

img_h = 656
img_w = 480


def parse_args():
    input_video_path = ROOT+'data\\data_in\\isld_karam_standing_577.avi'
    output_folder_path = ROOT+'data\\data_out\\'
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into a video.")
    parser.add_argument("-i", "--input_video_path", type=str, required=False, default=input_video_path) #default=ROOT+'\\data\\data_in\\'
    parser.add_argument("-o", "--output_folder_path", type=str, required=False, default=output_folder_path) #default =ROOT+'\\data\\data_out\\'
    parser.add_argument("-s", "--sample_interval", type=int, required=False,
                        default=1,
                        help="Sample every nth video frame to save to folder. Default 1.")
    parser.add_argument("-m", "--max_frames", type=int, required=False,
                        default=100000,
                        help="Max number of video frames to save to folder. Default 1e5")
    parser.add_argument("-f", "--info_file_path", type=str, required=False, default='')
    args = parser.parse_args()

    return args


class ReadFromVideo(object):
    def __init__(self, video_path, sample_interval=1):
        ''' A video reader class for reading video frames from video.
        Arguments:
            video_path
            sample_interval {int}: sample every kth image.
        '''
        if not os.path.exists(video_path):
            raise IOError("Video not exist: " + video_path)
        assert isinstance(sample_interval, int) and sample_interval >= 1
        self.cnt_imgs = 0
        self.is_stoped = False
        self.video = cv2.VideoCapture(video_path)
        ret, frame = self.video.read()
        self.next_image = frame
        self.sample_interval = sample_interval
        self.fps = self.get_fps()
        self.folders = []
        if not self.fps >= 0.0001:
            import warnings
            warnings.warn("Invalid fps of video: {}".format(video_path))

    def has_image(self):
        return self.next_image is not None

    def get_curr_video_time(self):
        imgs = 1.0 / self.fps * self.cnt_imgs
        return imgs

    def read_image(self):
        image = self.next_image
        for i in range(self.sample_interval):
            if self.video.isOpened():
                ret, frame = self.video.read()
                self.next_image = frame
            else:
                self.next_image = None
                break
        self.cnt_imgs += 1
        return image

    def stop(self):
        self.video.release()
        self.is_stoped = True

    def __del__(self):
        if not self.is_stoped:
            self.stop()

    def get_fps(self):

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        # With webcam get(CV_CAP_PROP_FPS) does not work.
        # Let's see for ourselves.

        # Get video properties
        if int(major_ver) < 3:
            fps = self.video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = self.video.get(cv2.CAP_PROP_FPS)
        return fps

class ImageDisplayer(object):
    def __init__(self):
        self._window_name = "images2video.py"
        cv2.namedWindow(self._window_name)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)


def main(args):
    video_loader = ReadFromVideo(args.input_video_path)

    if not args.output_folder_path:
        os.makedirs(args.output_folder_path)

    def set_output_filename(i):
        s = get_max_frame()
        return args.output_folder_path + str(s + 1) + ".jpg"
        # return "{:08d}".format(i) + ".jpg"

    def get_max_frame():
        if len(os.listdir(args.output_folder_path)) == 0:
            return 0
        else:
            f = []
            for filename in os.listdir(args.output_folder_path):
                val = int(filename.split('.')[0])
                f.append(val)
            s = max(f)
            return s

    # img_displayer = ImageDisplayer()
    cnt_img = 0
    # print(video_loader.get_curr_video_time())
    # print(video_loader.get_fps())
    # print(video_loader.get_curr_video_time())


    s = get_max_frame()
    for i in itertools.count():
        img = video_loader.read_image()
        if img is None:
            #print("Have read all frames from the video file.")
            break
        img = imutils.resize(img, width=img_w, height=img_h)
        if i % args.sample_interval == 0:
            cnt_img += 1
            # print("Processing {}th image".format(cnt_img))
            cv2.imwrite(set_output_filename(cnt_img), img)
            # img_displayer.display(img)
            if cnt_img == args.max_frames:
                print("Read {} frames. ".format(cnt_img) +
                      "Reach the max_frames setting. Stop.")
                break

    # fold_name = str(args.input_video_path.split('\\')[5].split('_')[2] + '_03')
    # if video_loader.folders:
    #     for i in video_loader.folders:
    #         if i == fold_name:
    #             continue
    #         else:
    #             video_loader.folders.append(fold_name)
    # else:
    #     video_loader.folders.append(fold_name)

    # f = open(args.info_file_path, 'r+')

    # with open(args.info_file_path, 'r+') as f:
    #     for cnt_line, line in enumerate(f):
    #         if line.find(str(args.input_video_path.split('\\')[5].split('_')[2] + '_03')) != -1:
    #             f.write('\n' + str(s+1) + ' ' + str(s+cnt_img))
    #         else:
    #             continue
    #     f.close()

    # for line in f:
    #     if line == str(args.input_video_path.split('\\')[5].split('_')[2] + '_03'):
    #         f.write('\n' + str(s+1) + ' ' + str(s+cnt_img))

    print(str(args.input_video_path.split('\\')[5].split('_')[2] + '_03') + '\n' + str(s+1) + ' ' + str(s+cnt_img))





    # with open(VALID_ING_TXT, "a") as file:
    #     for line in file:
    #         if ,,,,, in line:
    #             file.write()



if __name__ == "__main__":
    args = parse_args()
    assert args.sample_interval >= 0 and args.sample_interval
    main(args)
    print("Program stops: " + os.path.basename(__file__))

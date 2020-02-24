import os
import time
from os import path as osp

import numpy as np
import torch
import cv2
import pickle as pk
import csv
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize

import motmetrics as mm
from PIL import Image
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
# from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums

# Data transform
normalize_mean=[0.485, 0.456, 0.406]
normalize_std=[0.229, 0.224, 0.225]

def write_results(all_tracks, output_dir):
    """Write the tracks in the format for MOT16/MOT17 sumbission

    all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

    Each file contains these lines:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

    Files to sumbit:
    ./MOT16-01.txt
    ./MOT16-02.txt
    ./MOT16-03.txt
    ./MOT16-04.txt
    ./MOT16-05.txt
    ./MOT16-06.txt
    ./MOT16-07.txt
    ./MOT16-08.txt
    ./MOT16-09.txt
    ./MOT16-10.txt
    ./MOT16-11.txt
    ./MOT16-12.txt
    ./MOT16-13.txt
    ./MOT16-14.txt
    """

    #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"



    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = osp.join(output_dir, 'Video-result''.txt')

    with open(file, "w") as of:
        writer = csv.writer(of, delimiter=',')
        for i, track in all_tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])

# def main():

class Tracktor:
    def __init__(self, tracktor_config):
        self.tracktor_config = tracktor_config
        with open(tracktor_config, 'r') as f:
            self.tracktor = yaml.load(f)['tracktor']
            self.reid = self.tracktor['reid']

        # Set up seed 
        torch.manual_seed(self.tracktor['seed'])
        torch.cuda.manual_seed(self.tracktor['seed'])
        np.random.seed(self.tracktor['seed'])
        torch.backends.cudnn.deterministic = True

        # Output directory
        self.output_dir = osp.join(get_output_dir(self.tracktor['module_name']), self.tracktor['name'])
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # object detection
        self.obj_detect = FRCNN_FPN(num_classes=2)
        self.obj_detect.load_state_dict(torch.load(self.tracktor['obj_detect_model'],
                                   map_location=lambda storage, loc: storage))

        self.obj_detect.eval()
        self.obj_detect.cuda()

        # reid
        self.reid_network = resnet50(pretrained=False, **self.reid['cnn'])
        self.reid_network.load_state_dict(torch.load(self.tracktor['reid_weights'],
                                     map_location=lambda storage, loc: storage))
        self.reid_network.eval()
        self.reid_network.cuda()

        self.tracker = Tracker(self.obj_detect, self.reid_network, self.tracktor['tracker'])
        self.transforms = ToTensor()
        self.tracker.reset()

    def run(self, image):
        image = Image.fromarray(image[..., ::-1])
        image = self.transforms(image)[None, ...]

        blob = {"dets" : torch.Tensor([]), "img" : image}
        self.tracker.step(blob)

    def get_results(self):
        return self.tracker.get_results()

    def write_predictions(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        results = self.get_results()
        if self.tracktor['interpolate']:
            results = interpolate(results)

        print(f"Writing predictions to: {output_dir}")
        write_results(results, output_dir)

if __name__ == "__main__":
    tracktor_config = 'experiments/cfgs/tracktor.yaml'
    tracktor = Tracktor(tracktor_config)
    # tracktor_config = 'experiments/cfgs/tracktor.yaml'
    # with open(tracktor_config, 'r') as f:
    #     tracktor = yaml.load(f)['tracktor']
    #     reid = tracktor['reid']
    # torch.manual_seed(tracktor['seed'])
    # torch.cuda.manual_seed(tracktor['seed'])
    # np.random.seed(tracktor['seed'])
    # torch.backends.cudnn.deterministic = True

    # output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])

    # if not osp.exists(output_dir):
    #     os.makedirs(output_dir)
    # print(output_dir)
    # print(tracktor)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection

    # obj_detect = FRCNN_FPN(num_classes=2)
    # obj_detect.load_state_dict(torch.load(tracktor['obj_detect_model'],
    #                            map_location=lambda storage, loc: storage))

    # obj_detect.eval()
    # obj_detect.cuda()

    # reid
    # reid_network = resnet50(pretrained=False, **reid['cnn'])
    # reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
    #                              map_location=lambda storage, loc: storage))
    # reid_network.eval()
    # reid_network.cuda()

    # tracktor
    # if 'oracle' in tracktor:
    #     tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    # else:
    #     tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])
    # tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    # time_total = 0
    # num_frames = 0

    # # Data transform
    # normalize_mean=[0.485, 0.456, 0.406]
    # normalize_std=[0.229, 0.224, 0.225]
    # # dataset = Datasets(tracktor['dataset'])
    # transforms = ToTensor()
    # # transforms = Compose([ToTensor(), Normalize(normalize_mean,
    # #                                             normalize_std)])

    # tracker.reset()
    # # tracker.public_detections=False

    # start = time.time()

    # print(f"Tracking: video")

    # Load video and annotations
    cap = cv2.VideoCapture("/home/yc3390/camera_detection_demo/data/prid2011_videos/test_b_1min_1min.mp4")

    while True:
        ret, image = cap.read()
        if not ret:
            break

        tracktor.run(image)
        # # BGR to RGB
        # image = Image.fromarray(image[..., ::-1])
        # image = transforms(image)[None, ...]

        # # Detection
        # # if frame_count in gts.keys():
        # #     frames = 
        # blob = {"dets" : torch.Tensor([dets[i]]), "img" : image}
        # tracker.step(blob)
        # frame_count += 1
        # print("Finished ", frame_count, output_dir, image.shape)
        

    tracktor.write_predictions()
    # results = tracker.get_results()

    # time_total += time.time() - start


    # if tracktor['interpolate']:
    #     results = interpolate(results)

    # _log.info(f"Writing predictions to: {output_dir}")
    # write_results(results, output_dir)

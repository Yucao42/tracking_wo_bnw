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
from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
# ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


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

@ex.automain
def main(tracktor, reid, _config, _log, _run):
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector.")

    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    time_total = 0
    num_frames = 0
    mot_accums = []

    # Data transform
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]
    # dataset = Datasets(tracktor['dataset'])
    transforms = ToTensor()
    # transforms = Compose([ToTensor(), Normalize(normalize_mean,
    #                                             normalize_std)])

    tracker.reset()
    # tracker.public_detections=False

    start = time.time()

    _log.info(f"Tracking: video")

    # Load video and annotations
    cap = cv2.VideoCapture("/home/yc3390/camera_detection_demo/data/prid2011_videos/test_b_1min_1min.mp4")
    with open("/home/yc3390/camera_detection_demo/data/prid2011_videos/anno_b.pkl", 'rb') as f:
        gts = pk.load(f)

    det_file = "/data/yc3390/tracktor_output/output/tracktor/MOT17/Tracktor++/Video-result_ReID.txt"
    # with open("/data/yc3390/tracktor_output/output/tracktor/MOT17/Tracktor++/Video-result_ReID.pkl", 'rb') as f:
    #     dts = pk.load(f)

    #     for dt in dts:
    #         if len(dt['boxes'][0]):
    #             for i in range(len(dt['boxes'])):
    #                 dt['boxes'][i][-1] = -1
    offset = 25 * 60
    dets = {}
    for i in range(1, offset+1):
        dets[i] = []
    assert osp.exists(det_file)
    with open(det_file, "r") as inf:
        reader = csv.reader(inf, delimiter=',')
        for row in reader:
            x1 = float(row[2]) - 1
            y1 = float(row[3]) - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + float(row[4]) - 1
            y2 = y1 + float(row[5]) - 1
            score = float(row[6])
            bb = np.array([x1,y1,x2,y2], dtype=np.float32)
            dets[int(float(row[0]))].append(bb)
    frame_count = offset

    while True:
        ret, image = cap.read()
        if not ret:
            break
        # BGR to RGB
        image = Image.fromarray(image[..., ::-1])
        image = transforms(image)[None, ...]

        # Detection
        # if frame_count in gts.keys():
        #     frames = 
        blob = {"dets" : torch.Tensor([dets[i]]), "img" : image}
        tracker.step(blob)
        frame_count += 1
        print("Finished ", frame_count, output_dir, image.shape)
        
    results = tracker.get_results()

    time_total += time.time() - start

    _log.info(f"Tracks found: {len(results)}")
    _log.info(f"Runtime for video: {time.time() - start :.1f} s.")

    if tracktor['interpolate']:
        results = interpolate(results)

    if True:
        _log.info(f"No GT data for evaluation available.")
    else:
        mot_accums.append(get_mot_accum(results, seq))

    _log.info(f"Writing predictions to: {output_dir}")
    write_results(results, output_dir)

    # if tracktor['write_images']:
    #     plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(seq)))

    # _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
    #           f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")
    # if mot_accums:
    #     evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt], generate_overall=True)

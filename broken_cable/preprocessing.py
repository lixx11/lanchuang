#!/usr/bin/env python

import sys
import yaml
import time
from tqdm import tqdm

import numpy as np
import cv2

CAP_PROP_FPS = 5
CV_CAP_PROP_FRAME_COUNT = 7


def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, downsampling_size)
    frame = frame[crop_LU[1]:crop_LU[1]+crop_size[1],
                  crop_LU[0]:crop_LU[0]+crop_size[0]]  
    return frame 


if __name__ == '__main__':
    # parse configuration
    config_file = sys.argv[1]
    if len(sys.argv) == 3:
        module = sys.argv[2]  # use cv2 or skvideo
    else:
        module = 'cv2'
    print('reading video frames using %s' % module)
    print('loading configuration from %s' % config_file)
    config = yaml.load(open(config_file, 'r'))

    video_file = config['video_file']
    prefix = config['prefix']
    test_split = config['test_split']
    video_begin = time.strptime(config['begin'], 
        '%Y-%m-%d-%H-%M-%S')
    video_begin = time.mktime(video_begin)

    if module == 'cv2':
        reader = cv2.VideoCapture(video_file)
        nb_frames = int(reader.get(CV_CAP_PROP_FRAME_COUNT))
        fps = reader.get(CAP_PROP_FPS)
    elif module == 'skvideo':
        import skvideo.io as skio

        print('loading video using skvideo, this may take some time...')
        reader = skio.FFmpegReader(video_file)
        nb_frames = reader.inputframenum
        fps = reader.inputfps

    labels = np.ones(nb_frames, dtype=np.int16) * -1  # initialize label as -1
    time_in_secs = np.arange(nb_frames) / fps + video_begin
    train_indices = []
    test_indices = []
    # process annotation
    annotations = config['annotation']
    for annotation in config['annotation']:
        type_ = annotation['type']
        begin = time.strptime(annotation['begin'], 
            '%Y-%m-%d-%H-%M-%S')
        end = time.strptime(annotation['end'], 
            '%Y-%m-%d-%H-%M-%S')
        begin = time.mktime(begin)
        end = time.mktime(end)
        begin_frame = int((begin - video_begin) * fps)
        end_frame = int((end - video_begin) * fps)
        annotation['begin_frame'] = begin_frame
        annotation['end_frame'] = end_frame
        assert 0 <= begin_frame < nb_frames  # sanity check
        assert 0 <= end_frame < nb_frames

        labels[begin_frame:end_frame] = type_
        test_indices += np.random.choice(np.arange(begin_frame, end_frame),
            size=int(test_split*(end_frame-begin_frame)),
            replace=False).tolist()
    train_indices = list(set(np.where(labels>=0)[0]) - set(test_indices))
    test_indices.sort()
    train_indices.sort()
    # valid_indices = train_indices + test_indices

    # process video frames
    train_frames = []
    test_frames = []
    nb_train_dataset = 0
    nb_test_dataset = 0
    train_subset_ids = []  
    test_subset_ids = [] 
    downsampling_size = tuple(map(int, config['downsampling_size'].split('x')))
    max_frame_per_file = int(config['max_frame_per_file'])
    crop_size = map(int, config['crop_size'].split('x'))
    crop_LU = map(int, config['crop_LU'].split(','))
    skip_after = config['skip_after']
    if skip_after == -1:
        skip_after = np.inf
    
    if module == 'cv2':
        for i in tqdm(range(int(nb_frames))):
            if i > skip_after:
                break
            ret, frame = reader.read()
            frame = process_frame(frame)
            if i in train_indices:
                train_frames.append(frame)
                train_subset_ids.append(i)
                if len(train_frames) == max_frame_per_file:
                    np.savez('%s-train-%02d' % (prefix, nb_train_dataset),
                        frames=np.asarray(train_frames, dtype=np.uint8),
                        labels=labels[train_subset_ids],
                        time_in_secs=time_in_secs[train_subset_ids])
                    nb_train_dataset += 1
                    train_frames = []
                    train_subset_ids = []
            elif i in test_indices:
                test_frames.append(frame)
                test_subset_ids.append(i)
                if len(test_frames) == max_frame_per_file:
                    np.savez('%s-test-%02d' % (prefix, nb_test_dataset),
                        frames=np.asarray(test_frames, dtype=np.uint8),
                        labels=labels[test_subset_ids],
                        time_in_secs=time_in_secs[test_subset_ids])
                    nb_test_dataset += 1
                    test_frames = []
                    test_subset_ids = []
        cap.release()
    elif module == 'skvideo':
        for i,frame in enumerate(reader.nextFrame()):
            if i > skip_after:
                break
            print '\r%d/%d' % (i+1, nb_frames),
            frame = process_frame(frame)
            if i in train_indices:
                train_frames.append(frame)
                train_subset_ids.append(i)
                if len(train_frames) == max_frame_per_file:
                    np.savez('%s-train-%02d' % (prefix, nb_train_dataset),
                        frames=np.asarray(train_frames, dtype=np.uint8),
                        labels=labels[train_subset_ids],
                        time_in_secs=time_in_secs[train_subset_ids])
                    nb_train_dataset += 1
                    train_frames = []
                    train_subset_ids = []
            elif i in test_indices:
                test_frames.append(frame)
                test_subset_ids.append(i)
                if len(test_frames) == max_frame_per_file:
                    np.savez('%s-test-%02d' % (prefix, nb_test_dataset),
                        frames=np.asarray(test_frames, dtype=np.uint8),
                        labels=labels[test_subset_ids],
                        time_in_secs=time_in_secs[test_subset_ids])
                    nb_test_dataset += 1
                    test_frames = []
                    test_subset_ids = []

    # save last subset to file
    np.savez('%s-train-%02d' % (prefix, nb_train_dataset),
        frames=np.asarray(train_frames, dtype=np.uint8),
        labels=labels[train_subset_ids],
        time_in_secs=time_in_secs[train_subset_ids])

    np.savez('%s-test-%02d' % (prefix, nb_test_dataset),
        frames=np.asarray(test_frames, dtype=np.uint8),
        labels=labels[test_subset_ids],
        time_in_secs=time_in_secs[test_subset_ids])
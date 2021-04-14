import os
import sys
import pickle


import argparse
import numpy as np
from numpy.lib.format import open_memmap
from preprocess import pre_normalization
from utils.ntu_read_skeleton import read_xyz


training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2                        # 一个帧中最多能有两个骨架坐标
num_joint = 25                      # 节点个数是25
max_frame = 300                     #
toolbar_width = 30


def print_toolbar(rate, annotation = ''):
    """
    打印生成数据进度的信息
    :param rate:
    :param annotation:
    :return:
    """
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")


def gen_data(data_path,
             out_path,
             ignored_sample_path = None,
             benchmark = 'xview',
             part = 'eval',
             preprocess=False):

    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(filename[filename.find('A')+1:filename.find('A')+4])         # class the video beons to
        subject_id = int(filename[filename.find('P')+1:filename.find('P')+4])
        camera_id = int(filename[filename.find('C')+1:filename.find('C')+4])


        # Gets the training set partition
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'eval':
            issample = not istraining
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as fp:             # save label
        pickle.dump((sample_name, list(sample_label)), fp)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body), dtype=np.float32)

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(file=os.path.join(data_path, s),
                        max_body=max_body,
                        num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data

    if preprocess:
        fp = pre_normalization(fp)

    np.save('{}/{}_data.npy'.format(out_path, part), fp)
    end_toolbar()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='../../data/NTU-RGB-D/nturgb+d_skeletons'
    )
    parser.add_argument(
        '--out_folder', default='../../data/NTU-RGB-D'
    )
    parser.add_argument(
        '--ignored_sample', default='../../resources/NTU-RGB-D/samples_with_missing_skeletons.txt'
    )
    parser.add_argument("--preprocess", default=False, action='store_true', help='use preprocess or not')

    benchmark = ['xsub', 'xview']
    part = ['train', 'eval']
    arg = parser.parse_args()\

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gen_data(data_path=arg.data_path,
                     out_path=out_path,
                     ignored_sample_path=arg.ignored_sample,
                     benchmark=b,
                     part=p,
                     preprocess=arg.preprocess)

















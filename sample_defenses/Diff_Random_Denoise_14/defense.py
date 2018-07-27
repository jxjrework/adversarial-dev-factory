"""Sample Pytorch defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import argparse
import math
import numpy as np

from defense_random import defense_random
from defense_denoise_14 import defense_denoise_14

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR', default='',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE', default='',
                    help='Output file to save labels.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--itr_time', type=int, default=30, metavar='N',
                    help='Time of iteration for random padding (default: 30)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='Batch size (default: 16)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')



def main():
    start_time = time.time()

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print("Error: Invalid input folder %s" % args.input_dir)
        exit(-1)
    if not args.output_file:
        print("Error: Please specify an output file")
        exit(-1)
        
    no_gpu = args.no_gpu
    labels_denoise = defense_denoise_14(args.input_dir, args.batch_size, no_gpu)
    print(labels_denoise)

    labels_random = defense_random(args.input_dir, args.checkpoint_path, args.itr_time, args.batch_size)
    print(labels_random)

    print('diff filtering...')
    if (len(labels_denoise) == len(labels_random)):
        # initializing 
        final_labels = labels_denoise
        # Compare
        diff_index = [ii for ii in labels_denoise if labels_random[ii] != labels_denoise[ii]]
        if (len(diff_index) != 0):
            print(diff_index)
            for index in diff_index:
                final_labels[index] = 0
    else:
        print("Error: Number of labels returned by two defenses doesn't match")
        exit(-1)

    elapsed_time = time.time() - start_time
    print('elapsed time: {0:.0f} [s]'.format(elapsed_time))

    with open(args.output_file, 'w') as out_file:
        for filename, label in final_labels.items():
            out_file.write('{0},{1}\n'.format(filename, label))

if __name__ == '__main__':
    main()

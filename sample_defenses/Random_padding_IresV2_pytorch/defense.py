"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import argparse
import math
import random
import numpy as np

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from dataset import Dataset
import pretrainedmodels
from pretrainedmodels.models import pnasnet5large
from pretrainedmodels.models import inceptionresnetv2

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR', default='',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE', default='',
                    help='Output file to save labels.')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='Batch size (default: 16)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')
parser.add_argument('--itr', type=int, default=30, 
                    help='Number of iteration (default: 30)')

def batch_transform(inputs, transform, size):
    input_shape = list(inputs.size())
    res = torch.zeros(input_shape[0], input_shape[1], size, size)
    for i in range(input_shape[0]):
        res[i,:,:,:] = transform(inputs[i,:,:,:])
    return res

# codes for random padding
def padding_layer_iyswim(inputs, shape, transform):
    h_start = shape[0]
    w_start = shape[1]
    output_short = shape[2]
    # print(output_short)
    input_shape = list(inputs.size())
    #print(input_shape)
    # input shape (16, 3, 299, 299)
    input_short = min(input_shape[2:4])
    input_long = max(input_shape[2:4])
    #print(input_long, input_short)
    output_long = int(math.ceil( 1. * float(output_short) * float(input_long) / float(input_short)))
    output_height = output_long if input_shape[1] >= input_shape[2] else output_short
    output_width = output_short if input_shape[1] >= input_shape[2] else output_long  
    # print(output_height, output_width, output_long)
    padding = torch.nn.ConstantPad3d((w_start, output_width - w_start - input_shape[3], h_start, output_height - h_start - input_shape[2], 0,0), 0)
    outputs = padding(inputs)
    # print(type(outputs))
    return batch_transform(outputs, transform, 299)


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor

def main():
    start_time = time.time()
    args = parser.parse_args()

    tf = transforms.Compose([
        transforms.Resize([args.img_size,args.img_size]),
        transforms.ToTensor()
    ])

    tf_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])  

    tf_shrink = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([299,299]),
        transforms.ToTensor()
    ]) 
    
    with torch.no_grad():
        dataset = Dataset(args.input_dir, transform=tf)
        loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    #inceptionresnetv2
    model = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')
    model = model.cuda()
    model.eval()

    outputs = []
    for batch_idx, (input, _) in enumerate(loader):
        # print(input.size())
        length_input, _, _, _ = input.size()
        iter_labels = np.zeros([length_input, 1001, args.itr])
        for j in range(args.itr):
            # random fliping
            input0 = batch_transform(input, tf_flip, 299)
            # random resizing
            resize_shape_ = random.randint(310, 331)
            image_resize = 331
            tf_rand_resize = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([resize_shape_, resize_shape_]),
                transforms.ToTensor()
            ]) 
            input1 = batch_transform(input0, tf_rand_resize, resize_shape_)

            # ramdom padding
            shape = [random.randint(0, image_resize - resize_shape_), random.randint(0, image_resize - resize_shape_), image_resize]
            # print(shape)
       
            new_input = padding_layer_iyswim(input1, shape, tf_shrink)
            #print(type(new_input))
            if not args.no_gpu:
                new_input = new_input.cuda()
            with torch.no_grad():
                input_var = autograd.Variable(new_input)
                logits = model(input_var)
                labels = logits.max(1)[1]
                labels_index = labels.data.tolist() 
                print(len(labels_index))
                iter_labels[range(len(iter_labels)), labels_index, j] = 1
        final_labels = np.sum(iter_labels, axis=-1)
        labels = np.argmax(final_labels, 1)
        print(labels)
        outputs.append(labels)
    outputs = np.concatenate(outputs, axis=0)

    with open(args.output_file, 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, outputs):
            filename = os.path.basename(filename)
            out_file.write('{0},{1}\n'.format(filename, label))
    
    elapsed_time = time.time() - start_time
    print('elapsed time: {0:.0f} [s]'.format(elapsed_time))

if __name__ == '__main__':
    main()

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
import random
import numpy as np

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from dataset import Dataset
from res152_wide import get_model as get_model1
from inres import get_model as  get_model2
from v3 import get_model as get_model3
from resnext101 import get_model as get_model4

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR', default='',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE', default='',
                    help='Output file to save labels.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--checkpoint_path2', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='Batch size (default: 16)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')
parser.add_argument('--iteration', type=int, default=30, 
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

    if not os.path.exists(args.input_dir):
        print("Error: Invalid input folder %s" % args.input_dir)
        exit(-1)
    if not args.output_file:
        print("Error: Please specify an output file")
        exit(-1)
    
    tf = transforms.Compose([
           transforms.Resize([299,299]),
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
        mean_torch = autograd.Variable(torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1,3,1,1]).astype('float32')).cuda())
        std_torch = autograd.Variable(torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1,3,1,1]).astype('float32')).cuda())
        mean_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda())
        std_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda())

        dataset = Dataset(args.input_dir, transform=tf)
        loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
        config, resmodel = get_model1()
        config, inresmodel = get_model2()
        config, incepv3model = get_model3()
        config, rexmodel = get_model4()
        net1 = resmodel.net    
        net2 = inresmodel.net
        net3 = incepv3model.net
        net4 = rexmodel.net

    checkpoint = torch.load('denoise_res_015.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        resmodel.load_state_dict(checkpoint['state_dict'])
    else:
        resmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_inres_014.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        inresmodel.load_state_dict(checkpoint['state_dict'])
    else:
        inresmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_incepv3_012.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        incepv3model.load_state_dict(checkpoint['state_dict'])
    else:
        incepv3model.load_state_dict(checkpoint)
    
    checkpoint = torch.load('denoise_rex_001.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        rexmodel.load_state_dict(checkpoint['state_dict'])
    else:
        rexmodel.load_state_dict(checkpoint)

    if not args.no_gpu:
        inresmodel = inresmodel.cuda()
        resmodel = resmodel.cuda()
        incepv3model = incepv3model.cuda()
        rexmodel = rexmodel.cuda()
    inresmodel.eval()
    resmodel.eval()
    incepv3model.eval()
    rexmodel.eval()

    outputs = []
    iter = args.iteration
    # print(iter)
    for batch_idx, (input, _) in enumerate(loader):
        # print(input.size())
        length_input, _, _, _ = input.size()
        iter_labels = np.zeros([length_input, 1001, iter])
        for j in range(iter):
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
                input_tf = (input_var-mean_tf)/std_tf
                input_torch = (input_var - mean_torch)/std_torch

                labels1 = net1(input_torch,True)[-1]
                labels2 = net2(input_tf,True)[-1]
                labels3 = net3(input_tf,True)[-1]
                labels4 = net4(input_torch,True)[-1]

                labels = (labels1+labels2+labels3+labels4).max(1)[1] + 1  # argmax + offset to match Google's Tensorflow + Inception 1001 class ids

                labels_index = labels.data.tolist() 
                #if (len(labels_index) % args.batch_size != 0):
                #    zeros = [0]* (args.batch_size - len(labels_index) % args.batch_size)
                #    labels_index = labels_index + zeros
                print(len(labels_index))
                #iter_labels[range(len(iter_labels)),m, j] = 1 for m in labels_index
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

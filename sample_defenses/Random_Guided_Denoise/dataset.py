import torch.utils.data as data

from PIL import Image
import os
import re
import torch

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    if class_to_idx is None:
        class_to_idx = dict()
        build_class_idx = True
    else:
        build_class_idx = False
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        if build_class_idx and not subdirs:
            class_to_idx[label] = 0
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root,f))
                labels.append(label)
    if build_class_idx:
        classes = sorted(class_to_idx.keys(), key=natural_key)
        for idx, c in enumerate(classes):
            class_to_idx[c] = idx
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    if build_class_idx:
        return images_and_targets, classes, class_to_idx
    else:
        return images_and_targets


class Dataset(data.Dataset):

    def __init__(self, root, transform=None):
        imgs, classes, class_to_idx = find_images_and_targets(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=[]):
        if indices:
            return [self.imgs[i][0] for i in indices]
        else:
            return [x[0] for x in self.imgs]

# self-defined class to randomly resize and padding input iamges
class Rand_Dataset(data.Dataset):

    def __init__(self, root):
        imgs, classes, class_to_idx = find_images_and_targets(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')

        # random resizing and random padding
        resize_shape_ = random.randint(310, 331)
    image_resize = 331
    shape = [random.randint(0, image_resize - resize_shape_), random.randint(0, image_resize - resize_shape_), image_resize]
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
    tf_randomization = transforms.Compose([
           # random resizing
           transforms.Resize([resize_shape_, resize_shape_]),
           # left, top, right, bottom
           transforms.Pad((w_start, output_height - h_start - input_shape[2], output_width - w_start - input_shape[3], h_start))
           transforms.ToTensor()
    ])

        img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=[]):
        if indices:
            return [self.imgs[i][0] for i in indices]
        else:
            return [x[0] for x in self.imgs]
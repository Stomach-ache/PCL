from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
import augmentations
augmentations.IMAGE_SIZE = 32

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset):
    def __init__(self, dataset, ratio, noise_mode, root_dir, transform, mode, noise_file='', imb_type='exp', imb_ratio=1, imb_file=''):

        self.ratio = ratio # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise

        self.crop = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])

        if self.mode=='test':
            if dataset=='cifar10':
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.images = test_dic['data']
                self.images = self.images.reshape((10000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
                self.labels = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.images = test_dic['data']
                self.images = self.images.reshape((10000, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
                self.labels = test_dic['fine_labels']
        else:
            self.images=[]
            clean_label=[]
            if dataset=='cifar10':
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    self.images.append(data_dic['data'])
                    clean_label = clean_label+data_dic['labels']
                self.images = np.concatenate(self.images)
            elif dataset=='cifar100':
                train_dic = unpickle('%s/train'%root_dir)
                self.images = train_dic['data']
                clean_label = train_dic['fine_labels']
            self.images = self.images.reshape((50000, 3, 32, 32))
            self.images = self.images.transpose((0, 2, 3, 1))
            self.clean_label = np.array(clean_label)

            self.cls_num = 10 if dataset == 'cifar10' else 100

            self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_ratio)
            #imb_file = os.path.join(root_dir, 'cifar' + str(self.cls_num) + imb_type + '_' + str(imb_ratio))
            self.gen_imbalanced_data(self.img_num_list, imb_file)

            #noise_file = os.path.join(root_dir, 'cifar' + str(self.cls_num) + '_' + imb_type + '_' + str(imb_ratio) + '_' + noise_mode + '_' + str(ratio))
            self.get_noisy_data(self.cls_num, noise_file, noise_mode, ratio)

    def gen_imbalanced_data(self, img_num_per_cls, imb_file=None):
        if os.path.exists(imb_file):
            new_data = json.load(open(imb_file,"r"))
        else:
            new_data = []
            new_targets = []
            targets_np = np.array(self.clean_label, dtype=np.int64)
            classes = np.unique(targets_np)
            self.num_per_cls_dict = dict()
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = np.where(targets_np == the_class)[0]
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                new_data.extend(selec_idx)
            print ('saving imbalance data to %s ...' % imb_file)
            new_data = np.array(new_data).tolist()
            json.dump(new_data, open(imb_file, 'w'))
        new_data = np.array(new_data)
        self.images = self.images[new_data]
        self.clean_label = self.clean_label[new_data]

    def get_noisy_data(self, cls_num, noise_file, noise_mode, noise_ratio):
        train_label = self.clean_label

        if os.path.exists(noise_file):
            noise_label = json.load(open(noise_file,"r"))
        else:    #inject noise
            noise_label = []
            num_train = len(self.clean_label)
            idx = list(range(num_train))
            random.shuffle(idx)
            cls_num_list = self.img_num_list

            if noise_mode == 'sym':
                num_noise = int(noise_ratio * num_train)
                noise_idx = idx[:num_noise]

                for i in range(num_train):
                    if i in noise_idx:
                        newlabel = (random.randint(1, cls_num - 1) + train_label[i]) % cls_num
                        assert newlabel != train_label[i]
                        noise_label.append(newlabel)
                    else:
                        noise_label.append(train_label[i])

            elif noise_mode == 'imb':
                num_noise = int(noise_ratio * num_train)
                noise_idx = idx[:num_noise]

                p = np.array([cls_num_list for _ in range(cls_num)])
                for i in range(cls_num):
                    p[i][i] = 0
                p = p / p.sum(axis=1, keepdims=True)
                for i in range(num_train):
                    if i in noise_idx:
                        newlabel = np.random.choice(cls_num, p=p[train_label[i]])
                        assert newlabel != train_label[i]
                        noise_label.append(newlabel)
                    else:
                        noise_label.append(train_label[i])

            elif noise_mode == 'new':
                r = 1 - sum([(n / num_train) ** 2 for n in cls_num_list])
                num_noise = int(noise_ratio / r * num_train)
                noise_idx = idx[:num_noise]

                p = np.array([cls_num_list for _ in range(cls_num)])
                p = p / p.sum(axis=1, keepdims=True)
                for i in range(num_train):
                    if i in noise_idx:
                        newlabel = np.random.choice(cls_num, p=p[train_label[i]])
                        noise_label.append(newlabel)
                    else:
                        noise_label.append(train_label[i])

            noise_label = np.array(noise_label, dtype=np.int8).tolist()
            print("save noisy labels to %s ..." % noise_file)
            json.dump(noise_label, open(noise_file,"w"))

        self.labels = noise_label

        for c1, c0 in zip(self.labels, self.clean_label):
            if c1 != c0:
                self.img_num_list[c1] += 1
                self.img_num_list[c0] -= 1

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.clean_label) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls
  

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        if self.mode=='train':
            img = Image.fromarray(img)
            img_orig = self.transform(img)
            img_aug = self.crop(img)
            img_aug = augmentations.aug(img_aug,self.to_tensor)
            return img_orig, target, index, img_aug
        else:
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.images)


class cifar_dataloader():
    def __init__(self, dataset, ratio, noise_mode, batch_size, num_workers, root_dir, noise_file='', imb_type='exp', imb_ratio=1, imb_file=''):
        self.dataset = dataset
        self.ratio = ratio
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.imb_type = imb_type
        self.imb_ratio = imb_ratio
        self.imb_file = imb_file
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
        elif self.dataset=='cifar100':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
    def run(self):
        train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, ratio=self.ratio, root_dir=self.root_dir, transform=self.transform_train, mode="train",noise_file=self.noise_file, imb_type=self.imb_type, imb_ratio=self.imb_ratio, imb_file=self.imb_file)

        eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, ratio=self.ratio, root_dir=self.root_dir, transform=self.transform_test, mode="eval",noise_file=self.noise_file, imb_type=self.imb_type, imb_ratio=self.imb_ratio, imb_file=self.imb_file)

        test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, ratio=self.ratio, root_dir=self.root_dir, transform=self.transform_test, mode="test")

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=self.batch_size*4,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size*4,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)
        return train_loader,eval_loader,test_loader

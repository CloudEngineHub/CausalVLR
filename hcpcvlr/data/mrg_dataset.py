import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
from torch.utils.data import Dataset



class BaseDataset(Dataset):
    def __init__(self, cfgs, tokenizer, split, transform=None):
        self.image_dir = cfgs["image_dir"]
        self.ann_path = cfgs["ann_path"]
        self.max_seq_length = cfgs["max_seq_length"]
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        if cfgs["dataset_name"] == 'ffa_ir': self.dict2list4ffair()
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

    def dict2list4ffair(self):
        examples_list = []
        for k, v in self.examples.items():
            v['id'] = k
            v['image_path'] = v.pop('Image_path')
            v['report'] = v.pop('En_Report')
            examples_list.append(v)
        self.examples = examples_list


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class FFAIRDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        # image_path = eval(self.image_path[case_id])
        images = []
        count_img = len(image_path)
        for ind in range(count_img):
            # print("image_path[ind]",image_path[ind])
            image = Image.open(os.path.join(self.image_dir, image_path[ind])).convert('RGB')
            # im_shape = image.size  # (512, 512)

            # im_size_min = np.min(im_shape[0:2])
            # im_shapes.append(im_shape)
            # im_scales.append(self.height / im_size_min)
            if self.transform is not None:
                images.append(self.transform(image))

        images = torch.stack(images, 0)
        # im_info = [self.height, self.width, im_scales[0]]
        # im_info = torch.Tensor(im_info)

        reports_ids = example['ids']
        # print(reports_ids)
        reports_masks = example['mask']
        max_seq_length = len(reports_ids)

        """
        if count_img >= 96:
            # random select
            _index = np.arange(0, count_img)
            np.random.shuffle(_index)
            _index = _index[:96]
            images = images[_index]
        else:
            # all + random select
            _index = np.random.randint(0, count_img, 96 - count_img)
            images = torch.cat([images, images[_index]], 0)
        """
        if self.split == 'train':
            _index = torch.randint(0, len(image_path), [8])
            images = images[_index]
        else:
            if len(image_path) > 96:
                image_id = image_id[:96]
                images = images[:96]

        sample = (image_id, images, reports_ids, reports_masks, max_seq_length)
        return sample


class MixSingleImageDataset(Dataset):
    def __init__(self, cfgs, tokenizer, split, transform=None):
        self.image_dir = {'iu_xray': cfgs['image_dir']['iu_xray'],
                          'mimic_cxr': cfgs['image_dir']['mimic_cxr']}
        self.ann_path = {'iu_xray': cfgs['ann_path']['iu_xray'],
                          'mimic_cxr': cfgs['ann_path']['mimic_cxr']}
        self.max_seq_length = cfgs["max_seq_length"]
        self.split = split
        self.tokenizer = tokenizer  # vocab + <unk>
        self.transform = transform
        self.ann = {'iu_xray': json.loads(open(self.ann_path['iu_xray'], 'r').read()),
                    'mimic_cxr': json.loads(open(self.ann_path['mimic_cxr'], 'r').read())}

        self.examples = self.ann['iu_xray'][self.split]
        length = len(self.examples)
        for i in range(length):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'], dataset='iu_xray')[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

        self.examples += self.ann['mimic_cxr'][self.split]
        for i in range(length, len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'], dataset='mimic_cxr')[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        if len(image_path) > 1:
            if torch.rand(1) > 0.5:
                image = Image.open(os.path.join(self.image_dir['iu_xray'], image_path[0])).convert('RGB')
            else:
                image = Image.open(os.path.join(self.image_dir['iu_xray'], image_path[1])).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = Image.open(os.path.join(self.image_dir['mimic_cxr'], image_path[0])).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample




class MRGDataLoader(DataLoader):
    def __init__(self, cfgs, tokenizer, split, shuffle):
        self.cfgs = cfgs
        self.dataset_name = cfgs["dataset_name"]
        self.batch_size = cfgs["batch_size"]
        self.shuffle = shuffle
        self.num_workers = cfgs["num_workers"]
        self.tokenizer = tokenizer
        self.split = split

        # only ffa ir val and test set use 1 batch size
        if self.dataset_name == 'ffa_ir' and self.split != 'train':
            self.batch_size = 1

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.cfgs, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'mimic_cxr':
            self.dataset = MimiccxrSingleImageDataset(self.cfgs, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'ffa_ir':
            self.dataset = FFAIRDataset(self.cfgs, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'mix':
            self.dataset = MixSingleImageDataset(self.cfgs, self.tokenizer, self.split, transform=self.transform)
        else:
            raise ValueError

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)


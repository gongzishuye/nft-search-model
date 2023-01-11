from io import BytesIO
import glob
import os
import random

import PIL
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

FORMAL_LENGTH = 320


def resize_pad(img, resample=PIL.Image.BILINEAR,
               background_color=(0, 0, 0), LENGTH=FORMAL_LENGTH):
    """
    longer edge of the image will be matched to this LENGTH
    """
    w, h = img.size
    if w == h:
        resize_img = img.resize((LENGTH, LENGTH), resample=resample)
        return resize_img
    elif w > h:
        ratio = w / LENGTH
        resize_h = int(h / ratio)
        resize_img = img.resize((LENGTH, resize_h), resample=resample)
        background = Image.new(resize_img.mode, (LENGTH, LENGTH), background_color)
        up_left_idx = (LENGTH - resize_h) // 2
        background.paste(resize_img, (0, up_left_idx))
        return background
    elif w < h:
        ratio = h / LENGTH
        resize_w = int(w / ratio)
        resize_img = img.resize((resize_w, LENGTH), resample=resample)
        background = Image.new(resize_img.mode, (LENGTH, LENGTH), background_color)
        up_left_idx = (LENGTH - resize_w) // 2
        background.paste(resize_img, (up_left_idx, 0))
        return background


class Noise(object):
    """对原始图像加一些噪声
    """

    def __init__(self):
        self.crop_max_ratio = 0.1
        # select
        self.jitter_bright = transforms.ColorJitter(brightness=0.015)
        self.jitter_contrast = transforms.ColorJitter(contrast=0.015)
        self.jitter_saturation = transforms.ColorJitter(saturation=0.015)
        self.gaussian = transforms.GaussianBlur(15, 0.5)

        self.funcs = [self.jitter_bright, self.jitter_contrast, self.jitter_saturation,
                      self.gaussian, Noise.smooth, Noise.sharpen]

        # must
        self.resize_bilinear = transforms.Resize(FORMAL_LENGTH, InterpolationMode.BILINEAR)
        self.resize_bicubic = transforms.Resize(FORMAL_LENGTH, InterpolationMode.BICUBIC)

        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def _random_resize_pad(self, img):
        rand = random.random()
        if rand > 0.3:
            img_resize = resize_pad(img, PIL.Image.BILINEAR)
        elif 0.3 <= rand < 0.6:
            img_resize = resize_pad(img, PIL.Image.BICUBIC)
        else:
            img_resize = resize_pad(img, PIL.Image.NEAREST)
        return img_resize

    def _random_resize(self, img):
        rand = random.random()
        if rand > 0.5:
            img_resize = self.resize_bilinear(img)
        else:
            img_resize = self.resize_bicubic(img)
        return img_resize

    def _random_crop(self, img):
        w, h = img.size
        w_rand, h_rand = random.random(), random.random()
        w_crop = int((self.crop_max_ratio * w_rand) * w)
        h_crop = int((self.crop_max_ratio * h_rand) * h)

        w_coord_rand, h_coord_rand = random.random(), random.random()
        w_coord = int(w_coord_rand * w_crop)
        h_coord = int(h_coord_rand * h_crop)
        img_crop = img.crop((w_coord, h_coord, w_coord + w - w_crop, h_coord + h - h_crop))
        return img_crop

    def __call__(self, img):
        rand = random.random()
        if rand < 0.15:
            img = self._random_crop(img)

        w, h = img.size
        if w == h:
            img_resize = self._random_resize(img)
        else:
            img_resize = self._random_resize_pad(img)

        rand = random.random()
        if rand < 0.3:
            with BytesIO() as f:
                img_resize.save(f, format='JPEG', quality=92)
                f.seek(0)
                nosie_img = Image.open(f)
                func = random.choice(self.funcs)
                nosie_img = func(nosie_img)
                tensor_img = self.to_tensor(nosie_img)
                norm_img = self.norm(tensor_img)
                return norm_img
        elif 0.3 <= rand < 0.6:
            with BytesIO() as f:
                img_resize.save(f, format='webp', quality=92)
                f.seek(0)
                noise_img = Image.open(f)
                func = random.choice(self.funcs)
                noise_img = func(noise_img)
                tensor_img = self.to_tensor(noise_img)
                norm_img = self.norm(tensor_img)
                return norm_img
        else:
            func1 = random.choice(self.funcs)
            noise_img = func1(img_resize)
            func2 = random.choice(self.funcs)
            noise_img = func2(noise_img)
            tensor_img = self.to_tensor(noise_img)
            norm_img = self.norm(tensor_img)
            return norm_img

    @staticmethod
    def smooth(img):
        filter_img = img.filter(ImageFilter.SMOOTH)
        return filter_img

    @staticmethod
    def sharpen(img):
        filter_img = img.filter(ImageFilter.SHARPEN)
        return filter_img


class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, transform=None, noise=None):

        self.root_dir = root_dir
        self.transform = transform
        self.training_triplets = self.generate_triplets()
        self.noise = noise

    def generate_triplets(self):
        triplets = []
        collections = os.listdir(self.root_dir)
        print(f'{len(collections)} collections under {self.root_dir}')

        for collection in collections:
            nfts = glob.glob(os.path.join(self.root_dir, collection, '*'))
            try:
                nfts.remove(os.path.join(self.root_dir, collection, 'image_data'))
            except Exception:
                pass
            print(f'running {len(nfts)} under {collection}')

            for nft in nfts:
                neg_nft = random.choice(nfts)
                while neg_nft == nft:
                    neg_nft = random.choice(nfts)

                triplets.append([nft, neg_nft])
        return triplets

    def __getitem__(self, idx):

        nft, neg_nft = self.training_triplets[idx]

        anc_img = Image.open(nft).convert('RGB')

        pos_img = Image.open(nft).convert('RGB')
        pos_img = resize_pad(pos_img)

        neg_img = Image.open(neg_nft).convert('RGB')
        neg_img = resize_pad(neg_img)

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img
        }

        if self.transform:
            sample['anc_img'] = self.noise(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)


def get_dataloader(train_root_dir, valid_root_dir,
                   batch_size, num_workers):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])}

    noise = Noise()
    face_dataset = {
        'train': TripletFaceDataset(root_dir=train_root_dir,
                                    transform=data_transforms['train'],
                                    noise=noise),
        'valid': TripletFaceDataset(root_dir=valid_root_dir,
                                    transform=data_transforms['valid'],
                                    noise=noise)}

    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers)
        for x in ['train', 'valid']}

    data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}
    print('data_size', data_size)
    return dataloaders, data_size

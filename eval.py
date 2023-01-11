import torch
from torchvision import transforms
from models import FaceNetModel
from PIL import Image
import PIL
from torch.utils.data import Dataset


LENGTH = 320


def resize_pad(img, resample=PIL.Image.BILINEAR,
               background_color=(0, 0, 0)):
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


class ModelInfer(object):

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(checkpoint_path)
        self.transform = self.load_transforms()

    def load_model(self, checkpoint_path):
        model = FaceNetModel(pretrained=True)
        model.to(self.device)
        if not torch.cuda.is_available():
            model_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            model_dict = torch.load(checkpoint_path)
        model.load_state_dict(model_dict['state_dict'])
        model.eval()
        return model

    def load_transforms(self):
        trms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        return trms

    def __call__(self, img_paths):
        tensors = []
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            img = resize_pad(img)
            tensor = self.transform(img)
            tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        tensors = torch.cat(tensors, 0)
        logits = self.model(tensors.to(self.device))
        return logits


class MiniModelInfer(object):

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(checkpoint_path)
        self.model = self.model.to(self.device)

    def load_model(self, checkpoint_path):
        model = FaceNetModel(pretrained=True)
        model.to(self.device)
        if not torch.cuda.is_available():
            model_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            model_dict = torch.load(checkpoint_path)
        model.load_state_dict(model_dict['state_dict'])
        model.eval()
        return model

    def __call__(self, img_tensors):
        logits = self.model(img_tensors.to(self.device))
        return logits

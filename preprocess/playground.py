from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFilter


def transform_Resize():
    img = Image.open('data/InGroupCrypto.jpeg')
    img = img.convert("RGB")
    print(img.size)
    transform = transforms.Resize(400, InterpolationMode.BICUBIC)
    center_crop = transform(img)

    center_crop.save('result/InGroupCrypto.jpeg')


def transform_GaussianBlur():
    img = Image.open('data/InGroupCrypto.jpeg')
    img = img.convert("RGB")
    transform = transforms.Resize(400, InterpolationMode.BICUBIC)
    center_crop = transform(img)
    # 模糊半径越大, 正态分布标准差越大, 图像就越模糊
    transform_1 = transforms.GaussianBlur(11, 0.5)
    img_1 = transform_1(center_crop)
    # transform_2 = transforms.GaussianBlur(101, 10)
    # img_2 = transform_2(img)
    # transform_3 = transforms.GaussianBlur(101, 100)
    # img_3 = transform_3(img)
    img_1.save('result/InGroupCryptoGauss.jpeg')


def transform_ColorJitter():
    img = Image.open('data/InGroupCrypto.jpeg')
    img = img.convert("RGB")
    transform = transforms.Resize(400, InterpolationMode.BICUBIC)
    center_crop = transform(img)
    # 亮度设置为2
    transform_1 = transforms.ColorJitter(brightness=0.01)
    img_1 = transform_1(center_crop)
    img_1.save('result/InGroupCryptoJitter1.jpeg')
    # 对比度设置为2
    transform_2 = transforms.ColorJitter(contrast=0.01)
    img_2 = transform_2(center_crop)
    img_2.save('result/InGroupCryptoJitter2.jpeg')
    # # 饱和度设置为2
    transform_3 = transforms.ColorJitter(saturation=0.01)
    img_3 = transform_3(center_crop)
    img_3.save('result/InGroupCryptoJitter3.jpeg')


def transform_filter():
    img = Image.open('data/InGroupCrypto.jpeg')
    img = img.convert("RGB")
    print(img.size)
    transform = transforms.Resize(400, InterpolationMode.BICUBIC)
    center_crop = transform(img)
    # center_crop = center_crop.filter(ImageFilter.SMOOTH)
    center_crop = center_crop.filter(ImageFilter.SHARPEN)
    center_crop.save('result/InGroupCryptoFilter.jpeg')


def transform_back():
    img = Image.open('data/InGroupCrypto.jpeg')
    to_tensor = transforms.ToTensor()
    # img_tensor = to_tensor(img)
    img = img.convert("RGB")
    img_tensor1 = to_tensor(img)
    img.save('result/InGroupCryptoBack.jpeg')

    img1 = Image.open('result/InGroupCryptoBack.jpeg')
    img1_tensor1 = to_tensor(img1)
    return img1


# transform_back()

from io import BytesIO
import numpy as np

def convertToJpeg(ima):
    with BytesIO() as f:
        ima.save(f, format='JPEG')
        f.seek(0)
        ima_jpg = Image.open(f)
        img1_np = np.array(ima_jpg)
    return img1_np


img = Image.open('data/InGroupCrypto.jpeg')
img = img.convert("RGB")
img_np = np.array(img)
img1_np = convertToJpeg(img)
# img1_np = np.array(img1)
print(img1_np)
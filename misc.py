import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms  # 对PIL.Image进行变换的方法
from augmentation import HorizontalFlip
# import glob

NB_CLASSES = 128
IMAGE_SIZE = 224


class FurnitureDataset(Dataset):  # 继承了Dataset类
    def __init__(self, preffix: str, transform=None):
        self.preffix = preffix
        if preffix == 'val':
            path = 'validation'
        else:
            path = preffix
        path = f'./data/{path}.json'
        self.transform = transform
        # if preffix == 'test':
        #     img_idx = {int(p.name.split('.')[0])
        #                for p in Path(f'./{preffix}').glob('*.jpg')}
        # else:
        #     img_idx = {int(p.name.split('_')[0])
        #                for p in Path(f'./{preffix}').glob('*.jpg')}
        img_idx = {int(p.name.split('.')[0])  # class set {1,2,3,4,...}
                   for p in Path(f'./{preffix}').glob('*.jpg')}
        data = json.load(open(path))  # {dict} {'images':[{'url':...
        if 'annotations' in data:
            data = pd.DataFrame(data['annotations'])
        else:
            data = pd.DataFrame(data['images'])
        self.full_data = data  # data type:{DataFrame},是一个表格
        nb_total = data.shape[0]  # data.shape表格的行列数目,[0]表格的行数也就是json文件里面的总图片数
        data = data[data.image_id.isin(img_idx)].copy()  # img_idx是下载图片的集合，这是做一个交集
        # labels = {}
        #
        # def read_json():
        #     import json
        #     l = {}
        #     with open(path, 'rb') as f:
        #         o = json.load(f)
        #         for d in o['annotations']:
        #             l.update({d['image_id']: d['label_id']})
        #     return l

        #
        # if preffix == 'test':
        #     data['path'] = data.image_id.map(lambda i: "./{}/{}.jpg".format(preffix, i))
        # else:
        #     labels = read_json()
        #     data['path'] = data.image_id.map(lambda i:  "./{}/{}_{}.jpg".format(preffix, i, labels[i]))  # 验证的语法格式
        data['path'] = data.image_id.map(lambda i: "./{}/{}.jpg".format(preffix, i))  # Series.map(f) 应用元素级函数
        self.data = data
        print(f'[+] dataset `{preffix}` loaded {data.shape[0]} images from {nb_total}')

    def __len__(self):  # 返回数据集的大小
        return self.data.shape[0]

    def __getitem__(self, idx):  # 实现数据集的下标索引，返回对应的图像和标记
        row = self.data.iloc[idx]  # 索引DataFrame的第idx行,从0行开始,对应图片第一张
        img = Image.open(row['path'])
        if self.transform:
            img = self.transform(img)  # pre-process
        target = row['label_id'] - 1 if 'label_id' in row else -1  # why label_id - 1 ?
        return img, target


normalize = transforms.Normalize(  # 给定均值：(R,G,B) 方差：（R，G，B），即：Normalized_image=(image-mean)/std。
    mean=[0.485, 0.456, 0.406],    # 将会把Tensor正则化。
    std=[0.229, 0.224, 0.225]      # 即：Normalized_image=(image-mean)/std。
)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])
preprocess_hflip = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    HorizontalFlip(),
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    normalize               # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
])
preprocess_with_augmentation = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 切割中心点的位置随机选取。size可以是tuple也可以是Integer
    transforms.RandomHorizontalFlip(),  # 随机水平翻转给定的PIL.Image,概率为0.5
    transforms.ColorJitter(brightness=0.3, # 亮度
                           contrast=0.3, # 对比度
                           saturation=0.3), # 饱和度
    transforms.ToTensor(), # 转变成tensor
    normalize  # 正则化
])

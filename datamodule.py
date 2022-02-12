# 필요한 모듈들을 임포트합니다.
from torch.cuda import device_count
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from pytorch_lightning import LightningDataModule


# transform
def get_transform(train=True):
    if train:
        return T.Compose(
            [
                T.AutoAugment(),
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[.485, .456, .406],
                            std=[.229, .224, .225])
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[.485, .456, .406],
                            std=[.229, .224, .225])
            ]
        )


class CatsAndDogDataModule(LightningDataModule):
    def __init__(self,
                 directory='./cats_and_dogs_filtered',
                 batch_size=32):
        super(CatsAndDogDataModule, self).__init__()
        
        # dataset 선언
        self.train_dataset = ImageFolder(f'{directory}/train',
                                         transform=get_transform())
        self.validation_dataset = ImageFolder(f'{directory}/validation',
                                              transform=get_transform(train=False))
        
        self.batch_size = batch_size
        
        # num workers의 경우 학습 환경에 따라 다르게 설정해줘야 합니다.
        # 경험상 GPU의 갯수 * 4로 지정해주면 병목현상 등의 문제 없이 학습이 진행됩니다.
        # num workers 에 대한 자세한 설명은 아래 링크 참조해주세요
        # https://cvml.tistory.com/24
        self.num_workers = device_count() * 4
        
    # DataLoader 설정
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)
        
    def val_dataloader(self):
        return DataLoader(dataset=self.validation_dataset,
                          batch_size=self.batch_size * 2,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=False)

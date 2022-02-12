# 필요한 모듈들을 임포트합니다.
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet152

from collections import OrderedDict

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy


class ResnetTrainModule(LightningModule):
    def __init__(self,
                 freeze=True,
                 n_class=2):
        super(ResnetTrainModule, self).__init__()
        # 모델 선언부
        self.model = resnet152(pretrained=True)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        in_features = self.model.fc.in_features
                
        self.model.fc = nn.Sequential(
            OrderedDict(
                {
                    'batch-norm': nn.BatchNorm1d(in_features),
                    'activation': nn.ReLU(),
                    'fc': nn.Linear(in_features, n_class)
                }
            )
        )
        
        # loss function 선언
        self.loss_fn = nn.CrossEntropyLoss()
        self.freeze = freeze
        
        # validation에 활용할 지표
        self.acc = Accuracy()
    
    # optimizer 설정
    def configure_optimizers(self):
        # CNN 모델의 파라미터는 업데이트 하지 않고, fully connected layer만 업데이트
        return optim.Adam(self.model.fc.parameters(),
                          lr=1e-4)
        
    # 일반 PyTorch와 동일
    def forward(self, x):
        return self.model(x)
    
    # 코드의 중복을 피하기 위해 학습 루프를 함수로 선언
    def step(self, batch, train=True):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        if not train:
            self.acc(y_hat, y)
            
        return loss
    
    # 학습 및 검증 부분 선언
    def training_step(self, batch, *args, **kwargs):
        loss = self.step(batch)
        
        # logging
        # self.log('train_loss', loss, on_step=True, on_epoch=True)
        # 위와 같이 선언할 수도 있음 -> 결과물은 동일
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        
        # progress bar에 표시할 지표
        return {'loss': loss}
    
    
    def validation_step(self, batch, *args, **kwargs):
        loss = self.step(batch, train=False)
        
        # validation step에서는 step별 loss를 확인할 필요가 없음
        self.log('validation_loss', loss, on_step=False, on_epoch=True)
        self.log('Accuracy', self.acc, on_step=False, on_epoch=True)

# 필요한 모듈을 임포트합니다.
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from trainmodule import ResnetTrainModule
from datamodule import CatsAndDogDataModule

import warnings
warnings.filterwarnings('ignore')


# 작성해두었던 모듈들을 가져옵니다.
network = ResnetTrainModule()
datamodule = CatsAndDogDataModule()

# Logger를 선언합니다.
logger = TensorBoardLogger(
                           save_dir="classification_logs",  # 로그 파일들이 저장될 경로
                           name="cats&dogs",  # 로그 데이터의 이름
                           default_hp_metric=False,
                           )

# 체크포인트 콜백함수를 선언합니다.
checkpoint_callback = ModelCheckpoint(
                                     monitor='Accuracy',  # 어떤 지표를 기준으로 삼을 것인지
                                     dirpath='classification_ckpt',  # 체크포인트 파일들이 저장될 경로
                                     filename='cats&dogs_clssification',  # 체크포인트 파일의 이름(확장자 불요)
                                     mode='max'  # 위 지표가 최대일 때의 모델을 저장(min으로 설정 가능)
                                     )

# early stopping 콜백함수를 선언합니다.
early_stop_callback = EarlyStopping(
                                    monitor='Accuracy',  # 어떤 지표를 기준으로 삼을 것인지
                                    min_delta=1e-4,  # 위 지표가 얼마나 향상이 되어야 하는지
                                    patience=20,  # 몇 에포크동안 지켜볼 것인지
                                    mode='max'
                                    )


# trainer를 선언합니다.
trainer = Trainer(max_epochs=100,  # 최대 에포크
                  gpus=4,  # 학습에 사용할 GPU의 갯수
                  
                  # 보다 다양한 accelerator에 관련한 설명은 아래 링크 참조
                  # https://pytorch-lightning.readthedocs.io/en/stable/extensions/accelerators.html
                  accelerator='ddp',
                  callbacks=[early_stop_callback,
                             checkpoint_callback],  # 위에서 선언했던 콜백함수들
                  logger=logger  # 위에서 선언한 Logger
                  )

if __name__ == '__main__':
    # 학습 시작
    trainer.fit(network=network, datamodule=datamodule)
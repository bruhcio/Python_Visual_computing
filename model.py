import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td    # torch distribution 임포트

import lightning as L


#############################
class CVAE(L.LightningModule):              # 라이트닝 모듈 상속받아서 CVAE 클래스로 생성함
    def __init__(self, n_dim = 2, lr = 1e-3):   # 2차원으로
        super().__init__()
        self.n_dim = n_dim
        self.cond_dim = 10
        self.lr = lr

        # TODO
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(784 + self.cond_dim, 512),    #784 = 28 * 28이어서,
            nn.ReLU(),                              #리니어를 중간에서 한 번 꺾어줌
            nn.Linear(512, 256),                    #차원 줄임
            nn.ReLU()                               #비선형
        )
        self.fc_mu = nn.Linear(256, n_dim)          # 평균
        self.fc_var = nn.Linear(256, n_dim)         # 로그 분산 = 작년 비주얼 컴퓨팅에서 배운 거
        
        # 디코더 = 생각을 인코더의 역순이라고 생각하자. 인 - z - 디 - 생성
        self.decoder = nn.Sequential(
            nn.Linear(n_dim + self.cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, x, c):                    # x = 인풋 이미지, c = 조건임
        # TODO
        x_flattened = x.view(x.shape[0], -1)   # batch_size, 784  = 1차원으로 쭉 펼쳐서 784
        c = F.one_hot(c, num_classes = 10)      # batch_size, 10  = 원 핫 인코딩이라고 하는데
                                                # 딥러닝에서는 많이 사용된다고 함
                                                # 분류하는 경우에 범주형 데이터를 다룰 때 사용함.
                                                # 독립적으로 0번 데이터에 어쩌고, 3번 데이터에 저쩌고
                                                # 어쩌고와 저저고는 1,2,3,4 이런 것처럼 연속성은 없음
                                                # 이걸 만들어 주는 거
        xc = torch.cat([x_flattened, c], dim = 1) # 이미지랑 라벨을 붙임
        h = self.encoder(xc)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

        # TODO
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)          # 표준쳔차 계산
        sample = torch.randn_like(std)             # 표편 이용해서 샘플 생성
        return mu + sample * std

        # TODO
    def decode(self, z, c):                     # 인코더 역순
        c = F.one_hot(c, num_classes = 10)        # batch_size, 10
        zc = torch.cat([z, c], dim = 1)
        
        # 디코더 통과
        x_pred = self.decoder(zc)
        return x_pred.view(-1, 1, 28, 28)  # MNIST 이미지 형태로 변환

        # TODO
    def training_step(self, batch, batch_idx):  # 학습 순서
        x, c = batch                            # x = 이미지, c = 조건 - 선언
        mu, log_var = self.encode(x, c)         # 인
        z = self.reparameterize(mu, log_var)    # 샘
        x_pred = self.decode(z, c)              # 디
        batch_size = x.size(0)
        recon_loss = F.mse_loss(x_pred, x, reduction = 'sum') # 실제와 만든 거 비교
        kl_loss = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))
        loss = recon_loss + kl_loss                         # loss계산
        percent = (loss / (batch_size * 784)) * 100
        self.log('train_loss', percent,  prog_bar = True)       #v_num = 실험 버전 번호임
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) # 아담 사용 - 가장 범용적
        return optimizer

        # TODO
    def forward(self, x, c):  # 이미지랑 조건
        mu, log_var = self.encode(x, c)         # 인
        z = self.reparameterize(mu, log_var)    # 샘플화
        x_pred = self.decode(z, c)              # 디
        return x_pred
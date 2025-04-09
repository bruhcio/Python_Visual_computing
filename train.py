from dataset import train_loader
from model import CVAE

import lightning as L

model = CVAE()

# TODO max_epochs 값 조정 가능
trainer = L.Trainer(max_epochs = 20, accelerator="auto")        #에포크를 20으로 늘림
trainer.fit(model, train_dataloaders=train_loader)
trainer.save_checkpoint("./cvae.ckpt")

print(model.device)


from trainer import Trainer
from options import SignDetectOptions

options = SignDetectOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
    trainer.test()
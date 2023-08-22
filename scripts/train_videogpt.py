import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
sys.path.insert(0, "/home/jpatel_theaiinstitute_com/dev/VideoGPT/")
from videogpt import VideoGPT, VideoData

import wandb
from pytorch_lightning.loggers import WandbLogger

wandb.init(project="VideoGPT",name="VideoGPT_ucf101",
           settings=wandb.Settings(start_method="fork"))
wandb_logger = WandbLogger("VideoGPT_Logger")

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoGPT.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    args.class_cond_dim = data.n_classes if args.class_cond else None
    model = VideoGPT(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath="model_checkpoint",monitor='val/loss', mode='min', save_top_k=-1))

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus,
                      plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps, **kwargs,
                                            logger=wandb_logger)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()


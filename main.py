import os
import fire
from pytorch_lightning import Trainer

from lightning import Model
from util import init_exp_folder, Args
from lightning import (get_ckpt_callback, 
                       get_early_stop_callback,
                       get_logger)


def train(save_dir="./sandbox",
          exp_name="DemoExperiment",
          model="ResNet18",
          gpus=1,
          pretrained=True,
          num_classes=1,
          accelerator=None,
          gradient_clip_val=0.5,
          max_epochs=1,
          patience=10,
          limit_train_batches=1.0,
          tb_path="./sandbox/tb",
          loss_fn="BCE",
          weights_summary=None,
          ):
    """
    Run the training experiment.

    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        model: Model name
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        pretrained: Whether or not to use the pretrained model
        num_classes: Number of classes
        accelerator: Distributed computing mode
        gradient_clip_val:  Clip value of gradient norm
        limit_train_batches: Proportion of training data to use
        max_epochs: Max number of epochs
                patience: number of epochs with no improvement after
                                  which training will be stopped.
        tb_path: Path to global tb folder
        loss_fn: Loss function to use
        weights_summary: Prints a summary of the weights when training begins.

    Returns: None

    """
    args = Args(locals())
    init_exp_folder(args)
    m = Model(args)
    trainer = Trainer(gpus=gpus,
                      accelerator=accelerator,
                      logger=get_logger(save_dir, exp_name),
                      callbacks=[get_early_stop_callback(patience),
                                 get_ckpt_callback(save_dir, exp_name)],
                      weights_save_path=os.path.join(save_dir, exp_name),
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      weights_summary=weights_summary,
                      max_epochs=max_epochs)
    trainer.fit(m)


def test(ckpt_path,
         gpus=4,
         **kwargs):
    """
    Run the testing experiment.

    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
    Returns: None

    """
    m = Model.load_from_checkpoint(ckpt_path)
    trainer = Trainer(gpus=gpus)
    trainer.test(m)


if __name__ == "__main__":
    fire.Fire()

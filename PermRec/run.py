'''
Description: 
Author: Enming Yuan
email: yem19@mails.tsinghua.edu.cn
Date: 2021-08-15 16:27:19
LastEditTime: 2021-08-21 18:12:47
'''


from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from permrec.model import PermRecModel
from permrec.datamodule import PermRecDataModule


def cli_main():
    pl.seed_everything(42)

    cli = LightningCLI(PermRecModel, PermRecDataModule, subclass_mode_data=True)


if __name__ == '__main__':
    cli_main()

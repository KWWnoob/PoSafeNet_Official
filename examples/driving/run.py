import argparse
import importlib

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="show this help message and exit",
    )
    parser.add_argument("--model-module", type=str, required=True)
    parser.add_argument("--data-module", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="default")

    # 新增：训练随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for training",
    )

    # Import model
    model_mod = parser.parse_known_args()[0].model_module
    LitModel = importlib.import_module(
        f".{model_mod}",
        "models",
    ).LitModel
    parser = LitModel.add_model_specific_args(parser)

    # Import data module
    data_mod = parser.parse_known_args()[0].data_module
    LitDataModule = importlib.import_module(
        f".{data_mod}",
        "data_modules",
    ).LitDataModule
    parser = LitDataModule.add_data_specific_args(parser)

    # Add Trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if args.help:
        parser.print_help()
        return

    # 必须放在创建 model 和 datamodule 之前
    # 控制模型初始化、shuffle、dropout、NumPy、Python random 等
    pl.seed_everything(args.seed, workers=True)

    # Setup learning objects
    model = LitModel(**vars(args))
    datamodule = LitDataModule(args)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.default_root_dir,
        name=args.exp_name,
        default_hp_metric=True,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=args.every_n_epochs,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )

    # Run training
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
DPT-VQGAN 训练脚本（XCAT 心脏）
==============================================================
用法：
    python scripts/train_dpt_vqgan.py --train

说明：
    - 基于 configs/autoencoder/xcat_dpt_vqgan.yaml
    - 使用 ldm.data.xcat_seq_grouped 按被试分组返回 9 相位 batch
    - 输出日志：./logs/xcat-dpt-vqgan/
    - 支持 --resume 断点续训
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import *


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError("-n/--name and -r/--resume cannot be specified both.")

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")
        if os.path.isfile(opt.resume):
            logdir = "/".join(opt.resume.split("/")[:-2])
        else:
            logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        nowname = logdir.split("/")[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    if "gpus" in trainer_config and trainer_config["gpus"] == 1:
        trainer_config["accelerator"] = "dp"
    else:
        trainer_config["accelerator"] = "ddp"

    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)

    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    model = instantiate_from_config(config.model)

    # Logger
    default_logger_cfgs = {
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {"name": "testtube", "save_dir": logdir}
        },
    }
    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs = {"logger": instantiate_from_config(logger_cfg)}

    # Model checkpoint
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # Callbacks
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {
                "batch_frequency": 200,
                "max_images": 4,
                "clamp": True,
                "rescale": False,
            }
        },
        "learning_rate_logger": {
            "target": "main.LearningRateMonitor",
            "params": {"logging_interval": "step"}
        },
        "cuda_callback": {
            "target": "main.CUDACallback"
        },
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # Learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    ngpu = 1 if cpu else (len(str(trainer_config["gpus"]).split(',')) if isinstance(trainer_config["gpus"], str) else trainer_config["gpus"])
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(f"Setting learning rate to {model.learning_rate:.2e}")
    else:
        model.learning_rate = base_lr
        print(f"Setting learning rate to {model.learning_rate:.2e} (no scaling)")

    # USR1 for checkpoint
    def melk(*args, **kwargs):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    import signal
    signal.signal(signal.SIGUSR1, melk)

    if opt.train:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise

    print("Done.")

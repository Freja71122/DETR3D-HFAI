from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import hfai
import numpy as np
from mmcv import Config
from os import path as osp
from datasets import build_dataset
from models import build_model
from utils import get_root_logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from mmcv.runner import build_optimizer
from functools import partial
from datasets.collate import collate
from utils.scatter_gather import scatter_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    args = parser.parse_args()
    return args


def train(dataloader, model, optimizer, epoch, local_rank, start_step, best_acc):
    model.train()

    for step, batch in enumerate(dataloader):
        step += start_step
        batch, _ = scatter_kwargs(batch, {}, [local_rank])
        outputs = model.module.train_step(batch[0], optimizer)
        loss = outputs['loss']
        loss.backward()
        if local_rank == 0 and step % 20 == 0:
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}', flush=True)

        # 保存
        model.try_save(epoch, step + 1, others=best_acc)


def validate(dataloader, model, epoch, local_rank, optimizer):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch, _ = scatter_kwargs(batch, {}, [local_rank])
            outputs = model.module.val_step(batch[0], optimizer)
            loss = outputs['loss']

    if local_rank == 0:
        loss_val = loss.item() / dist.get_world_size() / len(dataloader)
        print(f'Epoch: {epoch}, Val Loss: {loss_val}', flush=True)

    return 1 / loss


def main(local_rank, args):
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # init distributed env first, since logger depends on the dist info.
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "2223")
    hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.environ.get("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    torch.cuda.set_device(local_rank)
    cfg.gpu_num = hosts * gpus
    # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
    cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpu_num / 8

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    logger_name = 'detr3d'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    train_dataset = build_dataset(cfg.data.train)

    train_datasampler = DistributedSampler(train_dataset, shuffle=True)
    # TODO: add shuffle
    train_dataloader = DataLoader(
        train_dataset, cfg.data.samples_per_gpu,
        sampler=train_datasampler,
        pin_memory=True,
        collate_fn=partial(collate, samples_per_gpu=cfg.data.samples_per_gpu),
    )
    if args.no_validate:
        val_data_cfg = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_data_cfg.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_data_cfg.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_data_cfg.test_mode = False
        val_dataset = build_dataset(val_data_cfg)
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        val_datasampler = DistributedSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            val_samples_per_gpu,
            sampler=val_datasampler,
            collate_fn=partial(collate, samples_per_gpu=val_samples_per_gpu),
            num_workers=cfg.data.workers_per_gpu
        )

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    seed = dist.get_rank()
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    model = DistributedDataParallel(
        model.cuda(), device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters
    )

    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    logger.info(f'Model:\n{model}')

    optimizer = build_optimizer(model, cfg.optimizer)
    ckpt_path = os.path.join(cfg.work_dir, 'latest.pt')
    start_epoch, start_step, best_acc = hfai.checkpoint.init(model, optimizer, ckpt_path=ckpt_path)
    best_acc = best_acc or 0

    for epoch in range(start_epoch, cfg.runner.max_epochs):
        # resume from epoch and step
        train_datasampler.set_epoch(epoch)
        train(train_dataloader, model, optimizer, epoch, local_rank, start_step, best_acc)
        start_step = 0  # reset

        if not args.no_validate:
            acc = validate(val_dataloader, model, epoch, local_rank, optimizer)
            # save
            if rank == 0 and local_rank == 0:
                if acc > best_acc:
                    best_acc = acc
                    print(f'New Best Acc: {100 * acc:.2f}%!')
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(cfg.work_dir, 'best.pt'))


if __name__ == '__main__':
    args = parse_args()
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(args,), nprocs=ngpus)

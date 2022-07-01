import platform
from mmcv.utils import Registry, build_from_cfg
from mmdet.datasets import DATASETS

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import CBGSDataset
    if cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset

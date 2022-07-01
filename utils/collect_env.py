import mmdet
import mmseg
from mmcv.utils import collect_env as collect_base_env


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMSegmentation'] = mmseg.__version__
    return env_info

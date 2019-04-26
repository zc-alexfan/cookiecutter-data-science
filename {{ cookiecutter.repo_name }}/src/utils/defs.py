from easydict import EasyDict as edict
import os

cfg = edict()
cfg.repo_path = os.path.abspath(os.path.join(__file__[:-19]))
cfg.log_path = os.path.join(cfg.repo_path, 'logs/')



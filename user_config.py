
import os.path as osp
from pathlib import Path
# Default neural network backend for each algo
# (Must be either 'tf1' or 'pytorch')
# DEFAULT_BACKEND = {
#     'vpg': 'pytorch',
#     'trpo': 'pytorch',
#     'ppo': 'pytorch',
#     'ddpg': 'pytorch',
#     'td3': 'pytorch',
#     'sac': 'pytorch'
# }

# Where experiment outputs are saved by default:
#DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')


# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching 
# experiments.
WAIT_BEFORE_LAUNCH = 5

def get_paths():
    project_path = Path('/home/jbrugger/PycharmProjects/lea_rl')
    paths = {
        'project_path' : project_path,
        'logger' : project_path / 'logger',
        'policies' : None ,# project_path / 'policies',
        'results' : project_path / 'results',
        'model_to_train' : None
    }
    return paths

import numpy as np
import torch
import logging
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.onlineLp.onlineLp import OnlineLp

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_onlineLpModel():
    onlineLp = OnlineLp("./data/traffic/")
    onlineLp.train("saved_model/onlineLpTest")


def run_onlineLp():
    """
    Run onlinelp model training and evaluation.
    """
    train_onlineLpModel()


if __name__ == '__main__':
    run_onlineLp()

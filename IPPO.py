import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append("./ma-gym")
from ma_gym.envs.combat.combat import Combat
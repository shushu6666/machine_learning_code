import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self,K):
        self.probs = np.random.uniform(size = K)
        self.best_dix = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_dix]
        self.K = K

    def step(self,k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂的伯努利赌博机，最优臂为%d，最优臂的成功概率为%.2f" % (K, bandit_10_arm.best_dix+1, bandit_10_arm.best_prob))
print(bandit_10_arm.probs)
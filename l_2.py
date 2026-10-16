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

class Slover:
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = 0

    def update_regret(self,k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
         raise NotImplementedError

    def run(self,num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Slover):
    def __init__(self,bandit,epsilon=0.01,init_prob=1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob]*self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0,self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r =self.bandit.step(k)
        self.estimates[k] += (r - self.estimates[k])/(self.counts[k]+1) 
        return k

if __name__ == "__main__":
    np.random.seed(1)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂的伯努利赌博机，最优臂为%d，最优臂的成功概率为%.2f" % (K, bandit_10_arm.best_dix+1, bandit_10_arm.best_prob))
    print(bandit_10_arm.probs)

import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils
import random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class BehaviorClone:
    # [FIX] device 从参数传入，不再依赖全局变量
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device):
        self.device = device
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, states, actions):
        states = torch.as_tensor(np.asarray(states), dtype=torch.float).to(self.device)
        # [FIX] 显式 long，避免 Windows 下 numpy int32 报错
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.long).view(-1, 1).to(self.device)
        log_probs = torch.log(self.policy(states).gather(1, actions))
        bc_loss = torch.mean(-log_probs)

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))


def test_agent(agent, env, n_episode):
    return_list = []
    for _ in range(n_episode):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)


class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device):
        self.device = device
        self.action_dim = action_dim            # [FIX] 保存动作维度，避免写死 2
        self.discriminator = Discriminator(state_dim, hidden_dim, action_dim).to(device)
        # [FIX] 属性名原来写成 self.discriminator.optimizer，导致 learn 里 AttributeError
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.as_tensor(np.asarray(expert_s), dtype=torch.float,
                                        device=self.device)
        agent_states = torch.as_tensor(np.asarray(agent_s), dtype=torch.float,
                                       device=self.device)
        # [FIX] 关键：显式 dtype=torch.long（Windows 上 numpy 默认 int32，不满足 one_hot 要求）
        expert_actions = torch.as_tensor(np.asarray(expert_a), dtype=torch.long,
                                         device=self.device)
        agent_actions = torch.as_tensor(np.asarray(agent_a), dtype=torch.long,
                                        device=self.device)

        expert_actions = F.one_hot(expert_actions, num_classes=self.action_dim).float()
        agent_actions = F.one_hot(agent_actions, num_classes=self.action_dim).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)

        # [FIX] 在 step 前先算 reward（detach），并加 eps 防止 log(0) = -inf
        rewards = -torch.log(agent_prob + 1e-8).detach().cpu().numpy().reshape(-1)

        bce = nn.BCELoss()
        discriminator_loss = bce(agent_prob, torch.ones_like(agent_prob)) + \
                             bce(expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones,
        }
        self.agent.update(transition_dict)


# [FIX] 改为显式传参，且 dtype 固定为 float32 / int64
def sample_expert_data(env, agent, n_episode):
    states = []
    actions = []
    for _ in range(n_episode):
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)


if __name__ == '__main__':
    n_expert_episode = 500        # 采样专家数据用的 episode 数
    n_gail_episode = 500          # [FIX] GAIL 训练轮数（原来被误设为 1）
    n_samples = 30                # 从专家数据中抽样数量

    actor_lr = 1e-3
    critic_lr = 1e-2
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    num_episodes = 250            # 训练专家 PPO 的 episode 数
    lr_d = 1e-3

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    try:
        env.seed(0)               # 旧版 gym
    except AttributeError:
        pass                      # 新版 gym 改用 env.reset(seed=0)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ---------- 1) 先训练专家策略 ----------
    ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    lmbda, epochs, eps, gamma, device)
    expert_return_list = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)
    print('专家策略平均回报:', np.mean(expert_return_list[-10:]))

    # ---------- 2) 用训练好的专家采样 ----------
    expert_s, expert_a = sample_expert_data(env, ppo_agent, n_expert_episode)
    random_index = random.sample(range(expert_s.shape[0]), n_samples)
    expert_s = expert_s[random_index]
    expert_a = expert_a[random_index]

    # ---------- 3) GAIL 训练 ----------
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)
    gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d, device)

    gail_return_list = []
    with tqdm(total=n_gail_episode, desc="GAIL Training") as pbar:
        for i in range(n_gail_episode):
            episode_return = 0
            state = env.reset()
            done = False
            state_list, action_list = [], []
            next_state_list, done_list = [], []

            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                done_list.append(done)
                state = next_state
                episode_return += reward

            gail_return_list.append(episode_return)
            gail.learn(expert_s, expert_a, state_list, action_list,
                       next_state_list, done_list)

            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(gail_return_list[-10:])})
            pbar.update(1)

    # ---------- 4) 绘图（专家 / GAIL 分开画） ----------
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(gail_return_list)), gail_return_list, label='GAIL')
    plt.axhline(y=np.mean(expert_return_list[-20:]), color='r',
                linestyle='--', label='Expert (avg)')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('GAIL on {}'.format(env_name))
    plt.legend()
    plt.show()
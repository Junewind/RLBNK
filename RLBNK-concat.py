#!/usr/bin/env python
# Time: 2021/3/10 下午9:47
# Author: Yichuan


# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/README.md 总共有320个star

"""

"""

import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import pickle

from ple import PLE
from ple.games.flappybird import FlappyBird

import random
from pandas.core.frame import DataFrame
from pgmpy.models import BayesianModel  # 用于模型构建
from pgmpy.estimators import BayesianEstimator  # 这里有好多esitimator的东西
from pgmpy.estimators import MaximumLikelihoodEstimator  # 用于参数学习
from pgmpy.inference import VariableElimination  # 用于推理

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#  #########################################################################


# 为了操作状态,使得它能够适应BN的输入
def state_mapping(num):
    # x
    if num[0] <= -0.15:
        x = 0  # N Negative
    elif -0.15 < num[0] <= 0.15:
        x = 1  # S Small
    else:
        x = 2  # P Positive
    # x_dot
    if num[1] <= -0.8:
        x_dot = 0  # N Negative
    elif -0.8 < num[1] <= 0.8:
        x_dot = 1  # S small
    else:
        x_dot = 2  # P Positive
    # theta
    if num[2] < -0.06:
        theta = 0  # N negative
    elif -0.06 <= num[2] <= 0.06:
        theta = 1  # S small
    else:
        theta = 2  # P positive
    # theta_dot
    if num[2] <= 0:
        theta_dot = 0  # N negative
    else:
        theta_dot = 2  # P positive

    return x, x_dot, theta, theta_dot


# 读取存储的专家数据
ENV_NAME = 'CartPole-v1'
demo_size = 2000
demo_data = ENV_NAME + "_" + str(demo_size) + ".pkl"
abs_path = "/1_CartPole_new/Orig_PPO/PPO_demo_traj/"
path = abs_path + demo_data
with open(path, "rb") as f:
    data = pickle.load(f)
s_list = data["s"]
a_list = data['a']
data = zip(s_list, a_list)

# 专家数据的格式转换
x_list = []
x_dot_list = []
theta_list = []
theta_dot_list = []
a_list = []

for num in data:
    x, x_dot, theta, theta_dot = state_mapping(num[0])
    # 数据放在列表里
    x_list.append(x)
    x_dot_list.append(x_dot)
    theta_list.append(theta)
    theta_dot_list.append(theta_dot)
    a_list.append(num[1])

# 构造pandas的DataFrame
final_data = {"x": x_list,
              "x_dot": x_dot_list,
              "theta": theta_list,
              "theta_dot": theta_dot_list,
              "a": a_list}
final_data = DataFrame(final_data)

# 构造贝叶斯网络结构
model = BayesianModel([("x", "a"),
                       ("x_dot", "a"),
                       ("theta", "a"),
                       ("theta_dot", "a")])
# 贝叶斯网络训练
model.fit(final_data, estimator=BayesianEstimator, prior_type="BDeu")  # default equivalent_sample_size=5
# 进行推理
model_infer = VariableElimination(model)


def infer(x, x_dot, theta, theta_dot):
    Q = model_infer.query(variables=['a'], evidence={'x': x, 'x_dot': x_dot, 'theta': theta, 'theta_dot': theta_dot},
                          show_progress=False)
    return Q.get_value(a=0), Q.get_value(a=1)

#  #########################################################################


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # PPO有两个policy，分别是self.policy和self.policy_old
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def train(seed):
    env_name = "CartPole-v0"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0] + 2
    action_dim = env.action_space.n

    max_episodes = 5000  

    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 2000  # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    #############################################

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()

        score = 0.0
        done = False

        while not done:
            timestep += 1
            x, x_dot, theta, theta_dot = state_mapping(state)
            Q0, Q1 = infer(x, x_dot, theta, theta_dot)
            cat_state = np.append(np.append(state, Q0), Q1)  # 合并之后的新的状态量

            # Running policy_old:
            action = ppo.policy_old.act(cat_state, memory)  
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal: 
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()  # ppo每update一次就clear一次memory
                timestep = 0

            score += reward

        print("episode :{}, avg score : {:.1f}".format(i_episode, score))
        # 保存训练过程奖励
        filename = 'Reward_log/data_' + str(seed) + '.txt'
        with open(filename, 'a+') as file:
            line = 'episode: %d  score: %d' % (i_episode, score) + '\n'
            file.writelines(line)


if __name__ == '__main__':
    for seed in range(10):
        train(seed)

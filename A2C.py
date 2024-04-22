import os
import sys
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from gameAPI.game import GamePacmanAgent
import config

''' 基于ResNet的策略和价值网络，用于处理图像输入 '''

class PolicyNet(torch.nn.Module):
    def __init__(self, config):
        super(PolicyNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 4)  # 假设有4种动作
        self.resnet = self.resnet.float()  # 确保模型是 Float

    def forward(self, x):
        x = x.float()  # 确保输入是 Float
        x = F.relu(self.resnet(x))
        return F.softmax(x, dim=1)

''' 价值网络，同样使用ResNet '''

class ValueNet(torch.nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 1)
        self.resnet = self.resnet.float()  # 确保模型是 Float

    def forward(self, x):
        x = x.float()  # 确保输入是 Float
        x = F.relu(self.resnet(x))
        return x

class ActorCriticAgent:
    def __init__(self, game_agent, config):
        self.game_agent = game_agent
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PolicyNet(config).to(self.device)
        self.critic = ValueNet().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def train(self):
        frames = []
        action_pred = None
        num_iter = 0

        while num_iter < self.config.max_train_iterations:
            frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action_pred)
            if is_gameover:
                self.game_agent.reset()

            # 对返回的帧进行预处理
            frame = self.preprocess_image(frame)
            frames.append(frame)

            if len(frames) == self.config.num_continuous_frames:
                image = np.concatenate(frames, axis=2)  # 沿着通道维度进行连接
                state = torch.from_numpy(image).unsqueeze(0).to(self.device)  # 添加一个批处理维度
                action_probs = self.actor(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
                formatted_action = self.formatAction(action, 'oriactionformat')

                frames.pop(0)  # 移除最旧的帧

                next_frame, _, next_is_gameover, next_reward, _ = self.game_agent.nextFrame(formatted_action)
                next_frame = self.preprocess_image(next_frame)
                next_state = np.concatenate([image[..., 3:], next_frame], axis=2)  # 移动帧并添加新帧
                next_state = torch.from_numpy(next_state).unsqueeze(0).to(self.device)

                td_target = reward + self.config.gamma * self.critic(next_state) * (1 - int(next_is_gameover))
                td_delta = td_target - self.critic(state)
                actor_loss = -(torch.log(action_probs.squeeze(0)[action]) * td_delta.detach())
                critic_loss = F.mse_loss(self.critic(state), td_target.detach())

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()


                self.log('[INFO] 训练迭代: {}, Actor Loss: {}, Critic Loss: {}'.format(num_iter, actor_loss.item(),
                                                                                           critic_loss.item()))

                num_iter += 1

            if num_iter % self.config.save_interval == 0:
                torch.save(self.actor.state_dict(), os.path.join(self.config.save_dir, 'actor_{}.pth'.format(num_iter)))
                torch.save(self.critic.state_dict(),
                           os.path.join(self.config.save_dir, 'critic_{}.pth'.format(num_iter)))

    def formatAction(self, action, outformat='networkformat'):
        # 模拟您的 DQN 中的 formatAction 方法
        if outformat == 'networkformat':
            return [0] * action + [1] + [0] * (4 - action - 1)
        elif outformat == 'oriactionformat':
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            return directions[action]

    def preprocess_image(self, image):
        return np.transpose(image / 255., (2, 0, 1))

    def log(self, message):
        print('{}: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), message))


game_agent = GamePacmanAgent(config)  # 创建游戏代理实例
ac_agent = ActorCriticAgent(game_agent, config)
ac_agent.train()

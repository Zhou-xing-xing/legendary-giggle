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


class ResNetPolicyNet(nn.Module):
    def __init__(self, config):
        super(ResNetPolicyNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, config.action_dim)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        return F.softmax(x, dim=1)


class ResNetValueNet(nn.Module):
    def __init__(self):
        super(ResNetValueNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        return x


class ActorCriticAgent():
    def __init__(self, game_agent, config):
        self.game_agent = game_agent
        self.config = config
        self.actor = ResNetPolicyNet(config).to(config.device)
        self.critic = ResNetValueNet().to(config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.memories = deque()

    def train(self):
        frames = []
        action_pred = None
        num_iter = 0

        while num_iter < self.config.max_train_iterations:
            frame, is_win, is_gameover, reward, action = self.game_agent.nextFrame(action=action_pred)

            if is_gameover:
                self.game_agent.reset()

            frames.append(frame)

            if len(frames) == self.config.num_continuous_frames:
                image = np.concatenate(frames, -1)  # 假设帧是图像数据
                image = self.preprocess_image(image)
                state = torch.from_numpy(image).unsqueeze(0).to(self.config.device)
                action_probs = self.actor(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action_pred = action_dist.sample().item()  # 从策略网络输出中抽样动作
                frames.pop(0)

                next_state, reward, done, _ = self.game_agent.step(action_pred)
                next_state = self.preprocess_image(next_state)
                next_state = torch.from_numpy(next_state).unsqueeze(0).to(self.config.device)

                # 更新actor-critic网络
                td_target = reward + self.config.gamma * self.critic(next_state) * (1 - int(done))
                td_delta = td_target - self.critic(state)
                actor_loss = -(torch.log(action_probs.squeeze(0)[action_pred]) * td_delta.detach())
                critic_loss = F.mse_loss(self.critic(state), td_target.detach())

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                if num_iter % self.config.log_interval == 0:
                    self.log('[INFO] 训练迭代: {}, Actor Loss: {}, Critic Loss: {}'.format(num_iter, actor_loss.item(),
                                                                                           critic_loss.item()))

                num_iter += 1

            if num_iter % self.config.save_interval == 0:
                torch.save(self.actor.state_dict(), os.path.join(self.config.save_dir, 'actor_{}.pth'.format(num_iter)))
                torch.save(self.critic.state_dict(),
                           os.path.join(self.config.save_dir, 'critic_{}.pth'.format(num_iter)))

    def preprocess_image(self, image):
        return np.transpose(image / 255., (2, 0, 1))  # 规范化并转置图像数据

    def log(self, message):
        print('{}: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), message))


class Config:
    def __init__(self):
        '''train'''
        self.gamma = 0.99
        self.batch_size = 32
        self.max_explore_iterations = 500
        self.max_memory_size = 5000
        self.max_train_iterations = 5000
        self.save_interval = 1000
        self.save_dir = 'model_saved'
        self.frame_size = None  # calculated automatically according to the layout file
        self.num_continuous_frames = 1
        self.logfile = 'train.log'
        self.use_cuda = torch.cuda.is_available()
        self.eps_start = 1.0  # prob to explore at first
        self.eps_end = 0.1  # prob to explore finally
        self.eps_num_steps = 10000

        '''test'''
        self.weightspath = os.path.join(self.save_dir, str(self.max_train_iterations) + '.pkl')  # trained model path

        '''game'''
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (255, 0, 255)
        self.SKYBLUE = (0, 191, 255)
        self.layout_filepath = 'layouts/mediumClassic.lay'  # decide the game map
        self.ghost_image_paths = [(each.split('.')[0], os.path.join(os.getcwd(), each)) for each in ['gameAPI/images/Blinky.png', 'gameAPI/images/Inky.png', 'gameAPI/images/Pinky.png', 'gameAPI/images/Clyde.png']]
        self.scaredghost_image_path = os.path.join(os.getcwd(), 'gameAPI/images/scared.png')
        self.pacman_image_path = ('pacman', os.path.join(os.getcwd(), 'gameAPI/images/pacman.png'))
        self.font_path = os.path.join(os.getcwd(), 'gameAPI/font/ALGER.TTF')
        self.grid_size = 32
        self.operator = 'ai'  # 'person' or 'ai', used in demo.py
        self.ghost_action_method = 'random'  # 'random' or 'catchup', ghost using 'catchup' is more intelligent than 'random'.

# 可以在此添加配置类或字典来管理诸如学习率、设备等参数。

# 示例使用
# config = Config()
# game_agent = YourGameAgent()
# ac_agent = ActorCriticAgent(game_agent, config)
# ac_agent.train()
config = Config()
game_agent = GamePacmanAgent(config)  # 创建游戏代理实例

ac_agent = ActorCriticAgent(game_agent, config)
ac_agent.train()
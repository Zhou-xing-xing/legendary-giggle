
import os
import sys
import time
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
from collections import deque


'''dqn'''

import torchvision.models as models


class DQNet(nn.Module):
	def __init__(self, config):
		super(DQNet, self).__init__()
		# 初始化一个预训练的 ShuffleNet 模型
		self.shufflenet = models.shufflenet_v2_x1_0(pretrained=True)

		# 修改第一个卷积层的输入通道数，以适应 num_continuous_frames * 3 个通道的输入
		first_conv = nn.Conv2d(config.num_continuous_frames * 3, 24, kernel_size=3, stride=2, padding=1, bias=False)
		self.shufflenet.conv1[0] = first_conv

		# 修改全连接层以输出动作的数量（在此示例中为动作空间大小）
		num_features = self.shufflenet.fc.in_features
		self.shufflenet.fc = nn.Linear(num_features, 4)  # 假定有4个动作

	def forward(self, x):
		return self.shufflenet(x)


'''dqn agent'''
class DQNAgent():
	def __init__(self, game_pacman_agent, dqn_net, config, **kwargs):
		self.game_pacman_agent = game_pacman_agent
		self.dqn_net = dqn_net
		self.config = config
		self.game_memories = deque()
		self.mse_loss = nn.MSELoss(reduction='elementwise_mean')

	'''train 方法用于训练 DQN 神经网络
		首先准备训练所需的环境和参数，然后进入训练循环。在循环中，根据当前游戏状态获取游戏帧、奖励、动作等信息，
		并根据探索策略或训练策略进行动作选择。同时，将游戏帧加入经验记忆队列，用于后续的训练。
		当经验记忆队列达到一定长度后，开始进行训练。在训练过程中，从经验记忆队列中随机采样得到训练数据，
		进行神经网络的前向传播、损失计算和反向传播等操作。根据训练策略选择动作，并记录日志信息。'''
	def train(self):
		# prepare
		if not os.path.exists(self.config.save_dir):
			os.mkdir(self.config.save_dir)
		if self.config.use_cuda:
			self.dqn_net = self.dqn_net.cuda()
		FloatTensor = torch.cuda.FloatTensor if self.config.use_cuda else torch.FloatTensor

		# 初始化参数
		frames = []
		optimizer = torch.optim.Adam(self.dqn_net.parameters(), lr=1e-3)
		num_iter = 0
		image = None
		image_prev = None
		action_pred = None
		score_best = 0
		num_games = 0
		num_wins = 0

		# 训练循环
		while True:
			if len(self.game_memories) > self.config.max_memory_size:
				self.game_memories.popleft()

			# 获取游戏帧、游戏状态等信息
			frame, is_win, is_gameover, reward, action = self.game_pacman_agent.nextFrame(action=action_pred)
			score_best = max(self.game_pacman_agent.score, score_best)

			# 游戏结束时重置游戏状态
			if is_gameover:
				self.game_pacman_agent.reset()
				if len(self.game_memories) >= self.config.max_explore_iterations:
					num_games += 1
					num_wins += int(is_win)

			# 将游戏帧加入帧队列
			frames.append(frame)

			# 当帧队列长度达到指定数量时，将帧队列转换为图片输入并加入经验记忆队列
			if len(frames) == self.config.num_continuous_frames:
				image_prev = image
				image = np.concatenate(frames, -1)
				exprience = (image, image_prev, reward, self.formatAction(action, outformat='networkformat'), is_gameover)
				frames.pop(0)
				if image_prev is not None:
					self.game_memories.append(exprience)

			# 探索阶段
			if len(self.game_memories) < self.config.max_explore_iterations:
				self.__logging('[STATE]: explore, [MEMORYLEN]: %d' % len(self.game_memories), self.config.logfile)

			# 训练阶段
			else:
				# 获取训练数据
				num_iter += 1
				images_input = []
				images_prev_input = []
				is_gameovers = []
				actions = []
				rewards = []
				for each in random.sample(self.game_memories, self.config.batch_size):
					image_input = each[0].astype(np.float32) / 255.
					image_input.resize((1, *image_input.shape))
					images_input.append(image_input)
					image_prev_input = each[1].astype(np.float32) / 255.
					image_prev_input.resize((1, *image_prev_input.shape))
					images_prev_input.append(image_prev_input)
					rewards.append(each[2])
					actions.append(each[3])
					is_gameovers.append(each[4])

				# 转换为 PyTorch 张量
				images_input_torch = torch.from_numpy(np.concatenate(images_input, 0)).permute(0, 3, 1, 2).type(FloatTensor)
				images_prev_input_torch = torch.from_numpy(np.concatenate(images_prev_input, 0)).permute(0, 3, 1, 2).type(FloatTensor)

				# 计算损失
				optimizer.zero_grad()
				q_t = self.dqn_net(images_input_torch).detach()
				q_t = torch.max(q_t, dim=1)[0]
				loss = self.mse_loss(torch.Tensor(rewards).type(FloatTensor) + (1 - torch.Tensor(is_gameovers).type(FloatTensor)) * (0.95 * q_t),
									 (self.dqn_net(images_prev_input_torch) * torch.Tensor(actions).type(FloatTensor)).sum(1))
				loss.backward()
				optimizer.step()

				# 制定动作决策
				prob = max(self.config.eps_start-(self.config.eps_start-self.config.eps_end)/self.config.eps_num_steps*num_iter, self.config.eps_end)
				if random.random() > prob:
					with torch.no_grad():
						self.dqn_net.eval()
						image_input = image.astype(np.float32) / 255.
						image_input.resize((1, *image_input.shape))
						image_input_torch = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(FloatTensor)
						action_pred = self.dqn_net(image_input_torch).view(-1).tolist()
						action_pred = self.formatAction(action_pred, outformat='oriactionformat')
						self.dqn_net.train()
				else:
					action_pred = None

				# 记录日志信息
				self.__logging('[STATE]: training, [ITER]: %d, [LOSS]: %.3f, [ACTION]: %s, [BEST SCORE]: %d, [NUMWINS/NUMGAMES]: %d/%d' % (num_iter, loss.item(), str(action_pred), score_best, num_wins, num_games), self.config.logfile)

				# 保存模型
				if num_iter % self.config.save_interval == 0 or num_iter == self.config.max_train_iterations:
					torch.save(self.dqn_net.state_dict(), os.path.join(self.config.save_dir, '%s.pkl' % num_iter))

				# 训练结束
				if num_iter == self.config.max_train_iterations:
					self.__logging('Train Finished!', self.config.logfile)
					sys.exit(-1)

	'''用于测试训练好的 DQN 模型在 Pacman 游戏中的表现。在测试过程中，模型根据当前游戏帧预测动作，并输出预测的动作值'''
	def test(self):
		# 如果使用 CUDA，将模型移至 GPU 上
		if self.config.use_cuda:
			self.dqn_net = self.dqn_net.cuda()

		# 设置模型为评估模式
		self.dqn_net.eval()

		# 根据是否使用 CUDA，选择相应的 Tensor 类型
		FloatTensor = torch.cuda.FloatTensor if self.config.use_cuda else torch.FloatTensor

		# 初始化帧列表和动作预测值
		frames = []
		action_pred = None
		while True:
			# 获取游戏帧、游戏状态、奖励、动作等信息
			frame, is_win, is_gameover, reward, action = self.game_pacman_agent.nextFrame(action=action_pred)
			# 如果游戏结束，重置游戏状态
			if is_gameover:
				self.game_pacman_agent.reset()

			# 将游戏帧加入帧列表
			frames.append(frame)

			# 当帧列表长度达到指定数量时
			if len(frames) == self.config.num_continuous_frames:

				# 将帧列表合并为一张图片
				image = np.concatenate(frames, -1)

				# 根据探索策略选择动作
				if random.random() > self.config.eps_end:
					with torch.no_grad():
						# 将图片输入模型进行预测，得到动作预测值
						image_input = image.astype(np.float32) / 255.
						image_input.resize((1, *image_input.shape))
						image_input_torch = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(FloatTensor)
						action_pred = self.dqn_net(image_input_torch).view(-1).tolist()
						action_pred = self.formatAction(action_pred, outformat='oriactionformat')
				else:
					action_pred = None
				# 移除帧列表中的第一帧
				frames.pop(0)
			print('[ACTION]: %s' % str(action_pred))

	'''将动作格式转换为网络格式或原始动作格式'''
	def formatAction(self, action, outformat='networkformat'):
		if outformat == 'networkformat':
			# left
			if action == [-1, 0]:
				return [1, 0, 0, 0]
			# right
			elif action == [1, 0]:
				return [0, 1, 0, 0]
			# up
			elif action == [0, -1]:
				return [0, 0, 1, 0]
			# down
			elif action == [0, 1]:
				return [0, 0, 0, 1]
			# error
			else:
				raise RuntimeError('something wrong in DQNAgent.formatAction')
		elif outformat == 'oriactionformat':
			idx = action.index(max(action))
			# left
			if idx == 0:
				return [-1, 0]
			# right
			elif idx == 1:
				return [1, 0]
			# up
			elif idx == 2:
				return [0, -1]
			# down
			elif idx == 3:
				return [0, 1]
			# error
			else:
				raise RuntimeError('something wrong in DQNAgent.formatAction')
		else:
			raise ValueError('DQNAgent.formatAction unsupport outformat %s...' % outformat)

	def __logging(self, message, savefile=None):
		content = '%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message)
		if savefile:
			f = open(savefile, 'a')
			f.write(content + '\n')
			f.close()
		print(content)


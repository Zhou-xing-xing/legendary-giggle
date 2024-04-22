"""
Function:
	define the game sprites.
"""
import pygame
import random


'''define the wall'''
class Wall(pygame.sprite.Sprite):
	def __init__(self, x, y, width, height, color, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.Surface([width, height])   # 创建一个表面对象，表示墙体的外观，尺寸为给定的宽度和高度
		self.image.fill(color)  # 用指定的颜色填充表面对象，以确定墙体的外观
		self.rect = self.image.get_rect()  # 获取表面对象的矩形区域，并将其赋值给实例变量self.rect

		# 设置墙体矩形区域的左上角坐标为给定的x和y，以确定墙体的初始位置
		self.rect.left = x
		self.rect.top = y


'''define the food'''
class Food(pygame.sprite.Sprite):
	def __init__(self, x, y, width, height, color, bg_color, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.Surface([width, height])
		self.image.fill(bg_color)
		self.image.set_colorkey(bg_color)
		pygame.draw.ellipse(self.image, color, [0, 0, width, height])
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)


'''define the ghost'''
class Ghost(pygame.sprite.Sprite):
	def __init__(self, x, y, role_image_path, scaredghost_image_path, image_size, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.ori_x, self.ori_y = x, y  # 记录鬼魂的初始位置
		self.role_name = role_image_path[0]  # 记录鬼魂角色名字

		# 加载受惊图片，并转换为游戏窗口所需的格式
		self.scared_image = pygame.image.load(scaredghost_image_path).convert()
		self.scared_image = pygame.transform.scale(self.scared_image, image_size)

		# 加载普通状态图片，并转换为游戏窗口所需的格式
		self.base_image = pygame.image.load(role_image_path[1]).convert()
		self.base_image = pygame.transform.scale(self.base_image, image_size)

		# 将普通状态图片设置为初始图片
		self.image = self.base_image.copy()

		# 获取鬼魂的矩形区域，并将其赋值给实例变量self.rect
		self.rect = self.image.get_rect()

		# 将鬼魂的初始位置设置为矩形区域的中心点
		self.rect.center = (x, y)

		# 记录鬼魂的上一个位置
		self.prev_x = x
		self.prev_y = y

		# 基础速度
		self.base_speed = [16, 16]

		# 当前速度
		self.speed = [0, 0]

		# 当前移动方向
		self.direction_now = None

		self.direction_legal = []
		self.directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
		self.is_scared = False
		self.is_scared_timer = 40  # 受惊状态持续时间
		self.is_scared_count = 0
		self.random_step_first = 100


	'''update'''
	def update(self, wall_sprites, gate_sprites, method='random', pacman_sprites=None):

		# 如果鬼魂处于随机移动初始阶段，则随机移动计数器减1，并设置移动方法为随机移动
		if self.random_step_first > 0:
			self.random_step_first -= 1
			method = 'random'

		if self.is_scared:
			self.base_speed = [8, 8]
			self.is_scared_count += 1
			if self.is_scared_count > self.is_scared_timer:
				self.is_scared_count = 0
				self.is_scared = False
		else:
			self.base_speed = [16, 16]

		# 获取鬼魂的移动方向，根据给定的方法和墙体、门精灵组等参数
		self.direction_now = self.__randomChoice(self.__getLegalAction(wall_sprites, gate_sprites), method, pacman_sprites)

		# 更新鬼魂的外观
		ori_image = self.base_image if not self.is_scared else self.scared_image
		if self.direction_now[0] < 0:
			self.image = pygame.transform.flip(ori_image, True, False)
		elif self.direction_now[0] > 0:
			self.image = ori_image.copy()
		elif self.direction_now[1] < 0:
			self.image = pygame.transform.rotate(ori_image, 90)
		elif self.direction_now[1] > 0:
			self.image = pygame.transform.rotate(ori_image, -90)

		# 根据移动方向和基础速度，计算鬼魂的速度
		self.speed = [self.direction_now[0] * self.base_speed[0], self.direction_now[1] * self.base_speed[1]]

		# 移动鬼魂
		self.rect.left += self.speed[0]
		self.rect.top += self.speed[1]
		return True
	'''reset'''
	def reset(self):
		self.random_step_first = 100
		self.rect.center = (self.ori_x, self.ori_y)
		self.is_scared_count = 0
		self.is_scared = False

	'''这段代码定义了鬼魂类中的 __getLegalAction() 方法，用于获取鬼魂在当前位置下可行的移动方向。
	方法遍历所有可能的移动方向，通过调用 __isActionLegal() 方法判断当前方向是否合法（即不会与墙体或门碰撞）。
	然后根据新得到的可行动方向列表与之前记录的可行动方向列表比较，如果两者相同，则返回当前移动方向；
	如果不同，则更新记录的可行动方向列表，并返回新的可行动方向列表'''
	def __getLegalAction(self, wall_sprites, gate_sprites):
		direction_legal = []
		for direction in self.directions:
			if self.__isActionLegal(direction, wall_sprites, gate_sprites):
				direction_legal.append(direction)
		if sorted(direction_legal) == sorted(self.direction_legal):
			return [self.direction_now]
		else:
			self.direction_legal = direction_legal
			return self.direction_legal


	'''方法首先检查移动方法是否为'random'，如果是，则随机从可行的移动方向列表中选择一个方向并返回。
	如果移动方法为'catchup'，则鬼魂会追逐吃豆人。方法会计算鬼魂到吃豆人的距离，并根据距离选择最佳的移动方向。
	如果鬼魂处于受惊状态，它会选择距离最远的方向；否则，它会选择距离最近的方向。
	然后根据选择的方向计算各个方向的选择概率，并根据概率随机选择一个方向作为移动方向，并返回该方向。'''
	def __randomChoice(self, directions, method='random', pacman_sprites=None):
		if method == 'random':
			return random.choice(directions)
		elif method == 'catchup':
			for pacman in pacman_sprites:
				pacman_pos = pacman.rect.center
			distances = []
			for direction in directions:
				# 计算鬼魂移动后的位置
				speed = [direction[0] * self.base_speed[0], direction[1] * self.base_speed[1]]
				ghost_pos = (self.rect.left+speed[0], self.rect.top+speed[1])

				# 计算鬼魂到吃豆人的曼哈顿距离，并将距离和移动方向添加到距离列表中
				distance = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
				distances.append([distance, direction])

			# 根据是否处于受惊状态选择最佳得分（距离）和概率
			if self.is_scared:
				best_score = max([d[0] for d in distances])
				best_prob = 0.8
			else:
				best_score = min([d[0] for d in distances])
				best_prob = 0.8

			# 获取距离最佳得分的所有方向
			best_directions = [d[1] for d in distances if d[0] == best_score]

			# 计算各方向的选择概率
			probs = {}
			for each in directions:
				probs[self.__formatDirection(each)] = (1 - best_prob) / len(directions)
			for each in best_directions:
				probs[self.__formatDirection(each)] += best_prob / len(best_directions)

			# 根据概率随机选择移动方向并返回
			total = float(sum(probs.values()))
			for key in list(probs.keys()):
				probs[key] = probs[key] / total
			r = random.random()
			base = 0.0
			for key, value in probs.items():
				base += value
				if r <= base:
					return self.__formatDirection(key)
		else:
			raise ValueError('Unsupport method %s in Ghost.__randomChoice...' % method)

	'''这个方法用于将方向的格式从字符串形式转换为坐标形式，或者从坐标形式转换为字符串形式'''
	def __formatDirection(self, direction):
		if isinstance(direction, str):
			directions_dict = {'left': [-1, 0], 'right': [1, 0], 'up': [0, -1], 'down': [0, 1]}
			direction = directions_dict.get(direction)
			if direction is None:
				raise ValueError('Error value %s in Ghost.__formatDirection...' % str(direction))
			else:
				return direction
		elif isinstance(direction, list):
			if direction == [-1, 0]:
				return 'left'
			elif direction == [1, 0]:
				return 'right'
			elif direction == [0, -1]:
				return 'up'
			elif direction == [0, 1]:
				return 'down'
			else:
				raise ValueError('Error value %s in Ghost.__formatDirection...' % str(direction))
		else:
			raise ValueError('Unsupport direction format %s in Ghost.__formatDirection...' % type(direction))

	'''用于检查给定方向的移动是否合法，即是否会与墙或门碰撞'''
	def __isActionLegal(self, direction, wall_sprites, gate_sprites):
		speed = [direction[0] * self.base_speed[0], direction[1] * self.base_speed[1]]
		x_prev = self.rect.left
		y_prev = self.rect.top
		self.rect.left += speed[0]
		self.rect.top += speed[1]
		is_collide = pygame.sprite.spritecollide(self, wall_sprites, False)
		if gate_sprites is not None:
			if not is_collide:
				is_collide = pygame.sprite.spritecollide(self, gate_sprites, False)
		self.rect.left = x_prev
		self.rect.top = y_prev
		return not is_collide


'''define the Pacman'''
class Pacman(pygame.sprite.Sprite):
	def __init__(self, x, y, role_image_path, image_size, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.role_name = role_image_path[0]
		self.base_image = pygame.image.load(role_image_path[1]).convert()
		self.base_image = pygame.transform.scale(self.base_image, image_size)
		self.image = self.base_image.copy()
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)
		self.base_speed = [16, 16]
		self.speed = [0, 0]
	'''update'''
	def update(self, direction, wall_sprites, gate_sprites):
		# update attributes

		# 根据移动方向更新精灵的图像
		if direction[0] < 0:
			# 向左移动，翻转图像
			self.image = pygame.transform.flip(self.base_image, True, False)
		elif direction[0] > 0:
			# 向右移动，不做处理
			self.image = self.base_image.copy()
		elif direction[1] < 0:
			# 向上移动，将图像逆时针旋转90度
			self.image = pygame.transform.rotate(self.base_image, 90)
		elif direction[1] > 0:
			# 向下移动，将图像顺时针旋转90度
			self.image = pygame.transform.rotate(self.base_image, -90)
		# 根据移动方向计算速度
		self.speed = [direction[0] * self.base_speed[0], direction[1] * self.base_speed[1]]
		# try move
		x_prev = self.rect.left
		y_prev = self.rect.top
		self.rect.left += self.speed[0]
		self.rect.top += self.speed[1]
		is_collide = pygame.sprite.spritecollide(self, wall_sprites, False)
		if gate_sprites is not None:
			if not is_collide:
				is_collide = pygame.sprite.spritecollide(self, gate_sprites, False)

		# 如果碰撞了，将精灵位置恢复到之前的位置
		if is_collide:
			self.rect.left = x_prev
			self.rect.top = y_prev
			return False
		return True
"""
功能：吃豆人游戏实现，包括游戏逻辑，界面展示，用户交互
"""

import cv2
import sys
import random
import pygame
import numpy as np
from .sprites import *

'''解析游戏地图布局文件'''
class LayoutParser:
    def __init__(self, config, **kwargs):
        self.gamemap = self.__parse(config.layout_filepath)  # 解析游戏地图
        self.height = len(self.gamemap)  # 获取地图高度
        self.width = len(self.gamemap[0])  # 获取地图宽度


    '''
    解析.lay文件
    .lay文件是一种文本格式的表示地图的文件
    该函数解析.lay文件，将其转化为游戏地图的表示形式
    返回列表数据结构
    '''
    def __parse(self, filepath):
        gamemap = []
        f = open(filepath)
        for line in f.readlines():
            elements = []
            for c in line:
                if c == '%':
                    elements.append('wall')
                elif c == '.':
                    elements.append('food')
                elif c == 'o':
                    elements.append('capsule')  # 能量豆
                elif c == 'P':
                    elements.append('Pacman')
                elif c in ['G']:
                    elements.append('Ghost')
                elif c == ' ':
                    elements.append(' ')
            gamemap.append(elements)
        f.close()
        return gamemap  # 返回游戏地图列表


'''定义游戏逻辑，界面展示'''
class GamePacmanAgent:
    def __init__(self, config, **kwargs):
        self.config = config
        self.layout = LayoutParser(config)
        self.screen_width = self.layout.width * config.grid_size  # 根据地图设置屏幕的尺寸
        self.screen_height = self.layout.height * config.grid_size
        self.reset()


    """
    根据地图信息创建并初始化游戏中的各种sprites对象，并将它们存储在不同的sprites组中
    pygame.sprite 是 Pygame 库中用于管理 2D 精灵（sprite）的模块。精灵是游戏中可见的图形对象
    """
    def __createGameMap(self):
        wall_sprites = pygame.sprite.Group()  # 创建了一个空的精灵组 wall_sprites，用于存储墙壁精灵对象
        pacman_sprites = pygame.sprite.Group()
        ghost_sprites = pygame.sprite.Group()
        capsule_sprites = pygame.sprite.Group()
        food_sprites = pygame.sprite.Group()

        ghost_idx = 0  # 初始化幽灵索引，用于标识不同幽灵的图像路径.用于区分不同幽灵的图像


        """
        循环遍历地图的每个格子
        根据地图中的不同元素类型，在相应的位置创建不同类型的精灵对象，并将它们添加到对应的精灵组中
        """
        for i in range(self.layout.height):
            for j in range(self.layout.width):
                elem = self.layout.gamemap[i][j]
                if elem == 'wall':
                    position = [j * self.config.grid_size, i * self.config.grid_size]
                    wall_sprites.add(Wall(*position, self.config.grid_size, self.config.grid_size, self.config.SKYBLUE))
                elif elem == 'food':
                    position = [j * self.config.grid_size + self.config.grid_size * 0.5,
                                i * self.config.grid_size + self.config.grid_size * 0.5]
                    food_sprites.add(Food(*position, 10, 10, self.config.GREEN, self.config.WHITE))
                elif elem == 'capsule':
                    position = [j * self.config.grid_size + self.config.grid_size * 0.5,
                                i * self.config.grid_size + self.config.grid_size * 0.5]
                    capsule_sprites.add(Food(*position, 16, 16, self.config.GREEN, self.config.WHITE))
                elif elem == 'Pacman':
                    position = [j * self.config.grid_size + self.config.grid_size * 0.5,
                                i * self.config.grid_size + self.config.grid_size * 0.5]
                    pacman_sprites.add(Pacman(*position, self.config.pacman_image_path,
                                              (self.config.grid_size, self.config.grid_size)))
                elif elem == 'Ghost':
                    position = [j * self.config.grid_size + self.config.grid_size * 0.5,
                                i * self.config.grid_size + self.config.grid_size * 0.5]
                    ghost_sprites.add(
                        Ghost(*position, self.config.ghost_image_paths[ghost_idx], self.config.scaredghost_image_path,
                              (self.config.grid_size, self.config.grid_size)))
                    ghost_idx += 1
        return wall_sprites, pacman_sprites, ghost_sprites, capsule_sprites, food_sprites


    '''
    更新游戏状态，绘制下一帧
    它处理玩家输入，更新吃豆人、食物、幽灵等游戏元素的状态，并判断游戏是否结束
    '''
    def nextFrame(self, action=None):
        if action is None:
            action = random.choice(self.actions)  # 如果没有指定动作就随机选择一个动作
        pygame.event.pump()  # 处理事件队列
        pressed_keys = pygame.key.get_pressed()  # 获取按键状态

        if pressed_keys[pygame.K_q]:  # 如果按下Q键，退出游戏
            sys.exit(-1)
            pygame.quit()

        is_win = False  # 初始化胜利标志
        is_gameover = False  # 初始化游戏结束标志
        reward = 0  # 初始化奖励值

        self.pacman_sprites.update(action, self.wall_sprites, None)  # 定义了pacman更新函数

        # 检测是否与食物或者能量豆发生了碰撞
        for pacman in self.pacman_sprites:
            food_eaten = pygame.sprite.spritecollide(pacman, self.food_sprites, True)
            capsule_eaten = pygame.sprite.spritecollide(pacman, self.capsule_sprites, True)

        nonscared_ghost_sprites = pygame.sprite.Group()  # 创建了一个空的非害怕状态的幽灵组
        dead_ghost_sprites = pygame.sprite.Group()

        # 如果吃掉害怕状态的幽灵会得到奖励值
        for ghost in self.ghost_sprites:
            if ghost.is_scared:
                if pygame.sprite.spritecollide(ghost, self.pacman_sprites, False):
                    reward += 6
                    dead_ghost_sprites.add(ghost)
            else:
                nonscared_ghost_sprites.add(ghost)

        # 被吃掉的幽灵将重置
        for ghost in dead_ghost_sprites:
            ghost.reset()
        del dead_ghost_sprites

        # 定义吃掉食物的奖励
        reward += len(food_eaten) * 2
        reward += len(capsule_eaten) * 3

        # 当 Pacman 吃到能量豆时，所有幽灵都会进入害怕状态
        if len(capsule_eaten) > 0:
            for ghost in self.ghost_sprites:
                ghost.is_scared = True

        self.ghost_sprites.update(self.wall_sprites, None, self.config.ghost_action_method, self.pacman_sprites)

        self.screen.fill(self.config.BLACK)  # 清空屏幕，准备在下一帧中重新绘制

        # 绘制sprites对象
        self.wall_sprites.draw(self.screen)
        self.food_sprites.draw(self.screen)
        self.capsule_sprites.draw(self.screen)
        self.pacman_sprites.draw(self.screen)
        self.ghost_sprites.draw(self.screen)

        # get frame
        frame = pygame.surfarray.array3d(pygame.display.get_surface())  # 获取当前游戏窗口的像素数据
        frame = cv2.transpose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 将Pygame 图像数据转换成 OpenCV 可用的图像格式

        # 将帧的大小被设置为固定值
        # self.config.frame_size = frame.shape[0], frame.shape[1], frame.shape[2]
        self.config.frame_size = (224, 224, 3)
        frame = cv2.resize(frame, self.config.frame_size[:2])

        # 显示得分
        self.score += reward

        # 绘制屏幕
        text = self.font.render('SCORE: %s' % self.score, True, self.config.WHITE)  # 字体渲染
        self.screen.blit(text, (2, 2))
        pygame.display.update()


        # 判断游戏是否结束
        if len(self.food_sprites) == 0 and len(self.capsule_sprites) == 0:
            is_win = True
            is_gameover = True
            reward = 10

        if pygame.sprite.groupcollide(self.pacman_sprites, nonscared_ghost_sprites, False, False):
            is_win = False
            is_gameover = True
            reward = -15

        if reward == 0:
            reward = -2
        return frame, is_win, is_gameover, reward, action

    '''用户控制的游戏逻辑
    def runGame(self):
        clock = pygame.time.Clock()  # 设置游戏帧率控制器
        is_win = False  # 初始化游戏状态为未胜利

        # 游戏主循环
        while True:
            # 退出循环条件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(-1)
                    pygame.quit()

            # 获取当前按键
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[pygame.K_UP]:
                self.pacman_sprites.update([0, -1], self.wall_sprites, None)
            elif pressed_keys[pygame.K_DOWN]:
                self.pacman_sprites.update([0, 1], self.wall_sprites, None)
            elif pressed_keys[pygame.K_LEFT]:
                self.pacman_sprites.update([-1, 0], self.wall_sprites, None)
            elif pressed_keys[pygame.K_RIGHT]:
                self.pacman_sprites.update([1, 0], self.wall_sprites, None)

            # 检测是否吃到食物和胶囊
            for pacman in self.pacman_sprites:
                food_eaten = pygame.sprite.spritecollide(pacman, self.food_sprites, True)
                capsule_eaten = pygame.sprite.spritecollide(pacman, self.capsule_sprites, True)

            # 创建未受惊吓的鬼魂组和已死亡的鬼魂组
            nonscared_ghost_sprites = pygame.sprite.Group()
            dead_ghost_sprites = pygame.sprite.Group()

            # 遍历所有鬼魂
            for ghost in self.ghost_sprites:
                if ghost.is_scared:
                    if pygame.sprite.spritecollide(ghost, self.pacman_sprites, False):
                        self.score += 6
                        dead_ghost_sprites.add(ghost)
                else:
                    nonscared_ghost_sprites.add(ghost)
            for ghost in dead_ghost_sprites:
                ghost.reset()

            self.score += len(food_eaten) * 2
            self.score += len(capsule_eaten) * 3

            if len(capsule_eaten) > 0:
                for ghost in self.ghost_sprites:
                    ghost.is_scared = True
            self.ghost_sprites.update(self.wall_sprites, None, self.config.ghost_action_method, self.pacman_sprites)
            self.screen.fill(self.config.BLACK)
            self.wall_sprites.draw(self.screen)
            self.food_sprites.draw(self.screen)
            self.capsule_sprites.draw(self.screen)
            self.pacman_sprites.draw(self.screen)
            self.ghost_sprites.draw(self.screen)

            # show the score
            text = self.font.render('SCORE: %s' % self.score, True, self.config.WHITE)
            self.screen.blit(text, (2, 2))
            # judge whether game over
            if len(self.food_sprites) == 0 and len(self.capsule_sprites) == 0:
                is_win = True
                break
            if pygame.sprite.groupcollide(self.pacman_sprites, nonscared_ghost_sprites, False, False):
                is_win = False
                break
            pygame.display.flip()

            clock.tick(10)
        if is_win:
            self.__showText(msg='You won!', position=(self.screen_width // 2 - 50, int(self.screen_height / 2.5)))
        else:
            self.__showText(msg='Game Over!', position=(self.screen_width // 2 - 80, int(self.screen_height / 2.5)))
    '''

    '''重置游戏的各种状态和元素，使游戏恢复到初始状态'''
    def reset(self):
        self.screen, self.font = self.__initScreen()
        self.wall_sprites, self.pacman_sprites, self.ghost_sprites, self.capsule_sprites, self.food_sprites = self.__createGameMap()
        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        self.score = 0

    """
    def step(self, action):
        # Convert action number to actual game action
        actual_action = self.actions[action]

        # Update game state based on action
        frame, is_win, is_gameover, reward, _ = self.nextFrame(actual_action)

        # Get new game state
        new_state = self.get_state()

        # Check if the game is over
        done = is_gameover
        return new_state, reward, done, {'is_win': is_win}

    def get_state(self):
        # Capture the current screen as the state
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = cv2.transpose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert from Pygame to OpenCV format
        frame = cv2.resize(frame, (84, 84))  # Resize to lower dimension for faster processing
        return frame
    '''show the game info'''
    """
    def __showText(self, msg, position):
        clock = pygame.time.Clock()
        text = self.font.render(msg, True, self.config.WHITE)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                    pygame.quit()
            self.screen.fill(self.config.BLACK)
            self.screen.blit(text, position)
            pygame.display.flip()
            clock.tick(10)



    '''initialize the game screen'''

    def __initScreen(self):
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        pygame.display.set_caption('')
        font = pygame.font.Font(self.config.font_path, 24)
        return screen, font



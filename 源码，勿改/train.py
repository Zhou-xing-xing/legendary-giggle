'''
Function:
	train the model
'''
import config
from nets.nets import DQNet, DQNAgent
from gameAPI.game import GamePacmanAgent


'''train the model'''
def train():
	game_pacman_agent = GamePacmanAgent(config)
	dqn_net = DQNet(config)
	dqn_agent = DQNAgent(game_pacman_agent, dqn_net, config)
	dqn_agent.train()






'''run'''
if __name__ == '__main__':
	train()
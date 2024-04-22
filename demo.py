'''
Function:
	show the effect of trained model used in game
'''
import torch
import config
from nets.nets import DQNet, DQNAgent
from gameAPI.game import GamePacmanAgent

'''run demo'''
def runDemo():
	if config.operator == 'ai':
		game_pacman_agent = GamePacmanAgent(config)
		dqn_net = DQNet(config)
		dqn_net.load_state_dict(torch.load(config.weightspath))
		dqn_agent = DQNAgent(game_pacman_agent, dqn_net, config)
		dqn_agent.test()
	elif config.operator == 'person':
		GamePacmanAgent(config).runGame()
	else:
		raise ValueError('config.operator should be <ai> or <person>...')

def runDemo():
	if config.operator == 'ai':
		game_pacman_agent = GamePacmanAgent(config)
		a2c_net = DQNet(config)
		a2c_net.load_state_dict(torch.load(config.weightspath))
		a2c_agent = DQNAgent(game_pacman_agent, a2c_net, config)
		a2c_agent.test()
	elif config.operator == 'person':
		GamePacmanAgent(config).runGame()
	else:
		raise ValueError('config.operator should be <ai> or <person>...')



'''run'''
if __name__ == '__main__':
	runDemo()
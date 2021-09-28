import numpy as np

def get_obs(obs,n_agent):
	for i in range(n_agent):
		index = np.zeros(n_agent)
		index[i] = 1
		obs[i] = np.hstack((obs[i],index))
	return obs

def test_agent(test_env, agents, n_ant):

	test_r, test_win = 0, 0
	for _ in range(20):
		test_env.reset()
		test_obs = get_obs(test_env.get_obs(), n_ant)
		test_mask = np.array([test_env.get_avail_agent_actions(i) for i in range(n_ant)])
		terminated = False
		while terminated == False:
			acts = []
			outs = agents.acting([np.array([test_obs[i]]) for i in range(n_ant)])
			for i in range(n_ant):
				a = np.argmax(outs[i][0] - 9e15*(1 - test_mask[i]))
				acts.append(a)
			reward, terminated, winner = test_env.step(acts)
			if winner.get('battle_won') == True:
				test_win += 1
			test_r += reward
			test_obs = get_obs(test_env.get_obs(),n_ant)
			test_mask = np.array([test_env.get_avail_agent_actions(i) for i in range(n_ant)])
	return test_r/20, test_win/20

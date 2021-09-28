import numpy as np
class ReplayBuffer(object):

	def __init__(self, buffer_size, state_space, obs_space, n_action, n_ant):
		self.buffer_size = buffer_size
		self.n_ant = n_ant
		self.pointer = 0
		self.len = 0
		self.states = np.zeros((self.buffer_size,state_space))
		self.actions = np.zeros((self.n_ant,self.buffer_size,n_action))
		self.masks = np.zeros((self.n_ant,self.buffer_size,n_action))
		self.next_masks = np.zeros((self.n_ant,self.buffer_size,n_action))
		self.rewards = np.zeros((self.buffer_size,1))
		self.next_states = np.zeros((self.buffer_size,state_space))
		self.dones = np.zeros((self.buffer_size,1))
		self.obs = np.zeros((self.n_ant,self.buffer_size,obs_space))
		self.next_obs = np.zeros((self.n_ant,self.buffer_size,obs_space))
		

	def getBatch(self, batch_size):

		index = np.random.choice(self.len, batch_size, replace=False)
		return self.obs[:,index], self.actions[:,index], self.rewards[index], self.next_obs[:,index], self.states[index], self.next_states[index], self.masks[:,index], self.next_masks[:,index], self.dones[index]

	def add(self, obs, action, reward, next_obs, state, next_state, mask, next_mask, done):

		self.obs[:,self.pointer] = obs
		self.actions[:,self.pointer] = action
		self.rewards[self.pointer] = reward
		self.next_obs[:,self.pointer] = next_obs
		self.dones[self.pointer] = done
		self.states[self.pointer] = state
		self.next_states[self.pointer] = next_state
		self.masks[:,self.pointer] = mask
		self.next_masks[:,self.pointer] = next_mask
		self.dones[self.pointer] = done
		self.pointer = (self.pointer + 1)%self.buffer_size
		self.len = min(self.len + 1, self.buffer_size)

	def getObs(self, batch_size):
		index = np.random.choice(self.len, batch_size, replace=False)
		index_positive = []
		for i in range(batch_size):
			t = np.random.randint(2,5)
			for j in range(1,t):
				k = index[i] - j
				if k < 0:
					break
				if self.dones[k] == True:
					break
			j -= 1
			index_positive.append(index[i] - j)
		index_positive = np.array(index_positive)
		return self.obs[:,index], self.obs[:,index_positive]
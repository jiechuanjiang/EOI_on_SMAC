import numpy as np
import sys,os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from model import *
from buffer import ReplayBuffer
from config import *
from utils import *
from smac.env import StarCraft2Env
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

n_try = float(sys.argv[1])

env = StarCraft2Env(map_name='so_many_baneling',move_amount = 0.3, range_=4.0)
env_info = env.get_env_info()
n_ant = env_info["n_agents"]
n_actions = env_info["n_actions"]
feature_space = env_info["obs_shape"]
state_space = env_info["state_shape"]
observation_space = feature_space + n_ant

buff = ReplayBuffer(capacity,state_space,observation_space,n_actions,n_ant)
agents = Agent(sess,observation_space,n_actions,state_space,n_ant,temperature)
eoi_net = intrisic_eoi(feature_space,n_ant)
get_eoi_reward = build_batch_eoi(feature_space,eoi_net,n_ant)

feature = np.zeros((batch_size,feature_space))
feature_positive = np.zeros((batch_size,feature_space))
q_critic = np.ones((batch_size*n_ant,n_actions))
o_critic = np.ones((batch_size*n_ant,observation_space))

agents.update()
env.reset()
obs = get_obs(env.get_obs(),n_ant)
state = env.get_state()
mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
terminated = False

f = open(sys.argv[1]+'_'+sys.argv[2]+'.txt','w')

while i_timestep<n_timestep:

	i_timestep += 1
	test_flag += 1
	epsilon = max(0.1, epsilon - 0.00002)#0.05

	action = []
	acts = []
	outs = agents.acting_exp([np.array([obs[i]]) for i in range(n_ant)]+[np.array([mask[i]]) for i in range(n_ant)])
	for i in range(n_ant):
		if np.random.rand() < epsilon:
			avail_actions_ind = np.nonzero(mask[i])[0]
			a = np.random.choice(avail_actions_ind)
		else:
			if epi%10 < n_try:
				a = np.argmax(outs[2*i][0])
			else:
				a = np.argmax(outs[2*i+1][0])
		action.append(to_categorical(a,n_actions))
		acts.append(a)

	reward, terminated, winner = env.step(acts)
	next_obs = get_obs(env.get_obs(),n_ant)
	next_state = env.get_state()
	next_mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
	buff.add(np.array(obs),action,reward,np.array(next_obs),state,next_state,mask,next_mask,terminated)
	obs = next_obs
	state = next_state
	mask = next_mask
	score += reward

	if terminated:
		epi += 1
		if test_flag > 10000:
			log_r, log_w = test_agent(env, agents, n_ant)
			h = str(log_r)+'	'+str(log_w)+'	 '+str(score/epi/10)
			f.write(h+'\n')
			f.flush()
			test_flag = 0
			score = 0
			epi = 0

		env.reset()
		obs = get_obs(env.get_obs(),n_ant)
		state = env.get_state()
		mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
		terminated = False

	if (i_timestep < 2000)|(i_timestep%2000!=0):
		continue
	
	for e in range(25):
		samples, positive_samples = buff.getObs(batch_size)
		feature_label = np.random.randint(0,n_ant,batch_size)
		for i in range(batch_size):
			feature[i] = samples[feature_label[i]][i][0:feature_space]
			feature_positive[i] = positive_samples[feature_label[i]][i][0:feature_space]
		sample_labels = to_categorical(feature_label,n_ant)
		positive_labels = eoi_net.predict(feature_positive,batch_size=batch_size)
		eoi_net.fit(feature,sample_labels+beta_1*positive_labels,batch_size=batch_size,epochs=1,verbose=0)

	for e in range(epoch):

		o, a, r, next_o, s, next_s, masks, next_masks, d = buff.getBatch(batch_size)
		
		q_intrinsic = agents.batch_q([o[i] for i in range(n_ant)])
		next_q_intrinsic = agents.batch_q_tar([next_o[i] for i in range(n_ant)])
		eoi_r = get_eoi_reward.predict([o[i][:,0:feature_space] for i in range(n_ant)],batch_size = batch_size)
		
		ind = 0
		for j in range(batch_size):
			for i in range(n_ant):
				q_intrinsic[i][j][np.argmax(a[i][j])] = gamma_1*(1-d[j])*next_q_intrinsic[i][j].max() + eoi_r[i][j]*10
				q_critic[ind] = q_intrinsic[i][j]
				o_critic[ind] = o[i][j]
				ind += 1
		agents.train_critics(o_critic, q_critic)

		q_target = agents.Q_tot_tar.predict([next_o[i] for i in range(n_ant)]+[next_masks[i] for i in range(n_ant)]+[next_s],batch_size = batch_size)
		q_target = r + q_target*gamma*(1-d)
		agents.train_qmix(o, a, s, q_target)

		agents.update_soft()

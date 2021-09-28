import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda, Input, Dense, Concatenate, Reshape, Add, Multiply, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
beta_2 = 0.2

def my_loss(y_true, y_pred):
	return categorical_crossentropy(y_true, y_pred) + beta_2*categorical_crossentropy(y_pred,y_pred)

def intrisic_eoi(dim, num_classes):

	In = Input(shape = (dim,))
	X = Dense(64, activation='relu')(In)
	X = Dense(64, activation='relu')(X)
	X = Dense(num_classes, activation='softmax')(X)
	model = Model(In,X)
	model.compile(loss=my_loss,optimizer=Adam(0.001))
	return model

def build_batch_eoi(dim, eoi_net, n_ant):

	In = []
	R = []
	for i in range(n_ant):
		In.append(Input(shape = (dim,)))
		R.append(Lambda(lambda x: K.reshape(x[:,i],(-1, 1)))(eoi_net(In[i])))
	
	model = Model(In,R)
	return model

def build_q_net(num_features,n_actions):

	O = Input(shape = (num_features,))
	h = Dense(64,activation = 'relu')(O)
	h = Dense(64,activation = 'relu')(h)
	V = Dense(n_actions)(h)
	model = Model(O,V)
	return model

def build_critic(num_features,n_actions):

	O = Input(shape = (num_features,))
	h = Dense(64,activation = 'relu')(O)
	h = Dense(64,activation = 'relu')(h)
	V = Dense(n_actions)(h)

	model = Model(O,V)
	return model

def build_mixer(n_ant,state_space):

	I1 = Input(shape = (n_ant,))
	I2 = Input(shape = (state_space,))

	W1 = Dense(n_ant*64)(I2)
	W1 = Lambda(lambda x: K.abs(x))(W1)
	W1 = Reshape((n_ant, 64))(W1)
	b1 = Dense(64)(I2)

	W2 = Dense(64)(I2)
	W2 = Lambda(lambda x: K.abs(x))(W2)
	W2 = Reshape((64, 1))(W2)
	b2 = Dense(1)(I2)

	h = Lambda(lambda x: K.batch_dot(x[0],x[1]))([I1,W1])
	h = Add()([h, b1])
	h = Activation('relu')(h)
	q_total = Lambda(lambda x: K.batch_dot(x[0],x[1]))([h,W2])
	q_total = Add()([q_total, b2])

	model = Model([I1,I2],q_total)
	return model

def build_Q_tot(obs_space,n_actions,state_space,n_ant,q_nets,mixer):

	O = []
	for i in range(n_ant):
		O.append(Input(shape = (obs_space,)))
	A = []
	for i in range(n_ant):
		A.append(Input(shape = (n_actions,)))
	S = Input(shape = (state_space,))

	q_values = []
	for i in range(n_ant):
		q_value = q_nets(O[i])
		q_values.append(Lambda(lambda x: K.reshape(K.sum(x,axis = 1),(-1,1)))(Multiply()([A[i],q_value])))
	q_values = Concatenate(axis=1)(q_values)
	q_total = mixer([q_values, S])

	model = Model(O+A+[S],q_total)
	return model

def build_Q_max(obs_space,n_actions,state_space,n_ant,q_nets,mixer):

	O = []
	for i in range(n_ant):
		O.append(Input(shape = (obs_space,)))
	M = []
	for i in range(n_ant):
		M.append(Input(shape = (n_actions,)))
	S = Input(shape = (state_space,))
	q_values = []
	for i in range(n_ant):
		q_value = Lambda(lambda x:x[0] - 9e15*(1 - x[1]))([q_nets(O[i]),M[i]])
		q_values.append(Lambda(lambda x: K.reshape(K.max(x,axis = 1),(-1,1)))(q_value))
	q_values = Concatenate(axis=1)(q_values)
	q_total = mixer([q_values, S])

	model = Model(O+M+[S],q_total)
	return model


def build_acting(num_features, actors, n_ant):

	Inputs = []
	for i in range(n_ant):
		Inputs.append(Input(shape = (num_features,)))

	actions = []
	for i in range(n_ant):
		actions.append(actors(Inputs[i]))

	return K.function(Inputs, actions)

def build_acting_explore(num_features, n_actions, actors, critic, n_ant):

	O = []
	M = []
	for i in range(n_ant):
		O.append(Input(shape = (num_features,)))
		M.append(Input(shape = (n_actions,)))

	A = []
	for i in range(n_ant):
		h1 = Lambda(lambda x:K.softmax(x[0] - 9e15*(1 - x[1]),axis = 1))([actors(O[i]),M[i]])
		h2 = Lambda(lambda x:K.softmax(x[0] - 9e15*(1 - x[1]),axis = 1))([critic(O[i]),M[i]])
		A.append(h1)
		A.append(h2)

	return K.function(O+M, A)

def build_batch_q(dim, critics, n_ant):

	O = []
	for i in range(n_ant):
		O.append(Input(shape = (dim,)))
	
	Q = []
	for i in range(n_ant):
		Q.append(critics([O[i]]))

	return K.function(O, Q)

class Agent(object):
	def __init__(self,sess,obs_space,n_actions,state_space,n_ant):
		super(Agent, self).__init__()
		self.sess = sess
		self.obs_space = obs_space
		self.n_actions = n_actions
		self.n_ant = n_ant
		self.state_space = state_space
		K.set_session(sess)
		
		self.q_nets = build_q_net(self.obs_space,self.n_actions)
		self.critics = build_critic(self.obs_space,self.n_actions)
		self.mixer = build_mixer(self.n_ant,self.state_space)
		self.Q_tot = build_Q_tot(self.obs_space,self.n_actions,self.state_space,self.n_ant,self.q_nets,self.mixer)
		self.acting = build_acting(self.obs_space, self.q_nets, self.n_ant)
		self.acting_exp = build_acting_explore(self.obs_space, self.n_actions, self.q_nets, self.critics, self.n_ant)
		self.batch_q = build_batch_q(self.obs_space,self.critics,self.n_ant)

		self.q_nets_tar = build_q_net(self.obs_space,self.n_actions)
		self.critics_tar = build_critic(self.obs_space,self.n_actions)
		self.mixer_tar = build_mixer(self.n_ant,self.state_space)
		self.Q_tot_tar = build_Q_max(self.obs_space,self.n_actions,self.state_space,self.n_ant,self.q_nets_tar,self.mixer_tar)
		self.batch_q_tar = build_batch_q(self.obs_space,self.critics_tar,self.n_ant)
		
		self.label = tf.placeholder(tf.float32,[None, 1])
		self.label_critic = tf.placeholder(tf.float32,[None, n_actions])
		
		self.optimize = []
		self.optimize.append(tf.train.AdamOptimizer(0.0005).minimize(tf.reduce_mean((self.label - self.Q_tot.output)**2), var_list = self.q_nets.trainable_weights))
		self.optimize.append(tf.train.AdamOptimizer(0.0005).minimize(tf.reduce_mean((self.label - self.Q_tot.output)**2), var_list = self.mixer.trainable_weights))
		self.opt_qmix = tf.group(self.optimize)

		self.opt_critic = tf.train.AdamOptimizer(0.0005).minimize(tf.reduce_mean((self.label_critic - self.critics.outputs[0])**2), var_list = self.critics.trainable_weights)

		self.sess.run(tf.global_variables_initializer())

	def train_qmix(self, O, A, S, label):

		dict1 = {}
		for i in range(self.n_ant):
			dict1[self.Q_tot.inputs[i]] = O[i]
			dict1[self.Q_tot.inputs[i+self.n_ant]] = A[i]

		dict1[self.Q_tot.inputs[2*self.n_ant]] = S			
		dict1[self.label] = label
		self.sess.run(self.opt_qmix, feed_dict=dict1)

	def train_critics(self, X, label):

		dict1 = {}
		dict1[self.critics.inputs[0]] = X
		dict1[self.label_critic] = label
		self.sess.run(self.opt_critic, feed_dict=dict1)

	def update(self):
		self.Q_tot_tar.set_weights(self.Q_tot.get_weights())
		self.critics_tar.set_weights(self.critics.get_weights())

	def update_soft(self):

		weights = self.Q_tot.get_weights()
		target_weights = self.Q_tot_tar.get_weights()
		for w in range(len(weights)):
			target_weights[w] = (1 - 0.99)* weights[w] + 0.99* target_weights[w]
		self.Q_tot_tar.set_weights(target_weights)

		weights = self.critics.get_weights()
		target_weights = self.critics_tar.get_weights()
		for w in range(len(weights)):
			target_weights[w] = (1 - 0.99)* weights[w] + 0.99* target_weights[w]
		self.critics_tar.set_weights(target_weights)

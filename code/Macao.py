__author__ = 'arthur'

import numpy as np

class Graph:
	def __init__(self, cluster_size, r_mat):
		self.cluster_size = self.cluster_size
		self.num_cluster = np.size(self.cluster_size)
		self.cluster_bounds = np.zeros(self.num_cluster+1)
		self.cluster_bounds[1:self.num_cluster+1] = np.cumsum(cluster_size)
		self.N = self.cluster_bounds[self.num_cluster]
		self.nodes = np.arange(self.N)
		self.C = []
		for i in range(0,self.num_cluster):
			self.C.append(self.nodes[self.cluster_bounds[i]:self.cluster_bounds[i+1]])
		self.r_mat = r_mat

	def find_cluster(self, I):
		return np.searchsorted(self.cluster_bounds,I)-1

	def observe(self, I):
		c = self.find_cluster(I)
		obs = np.zeros(self.N)
		for i in range(0,self.num_cluster):
			obs[self.cluster_bounds[i]:self.cluster_bounds[i+1]] = np.random.choice(2, size=self.cluster_size[i], p =[1-self.r_mat[c,i], self.r_mat[c,i]])
		obs[I] = 1
		return obs[I]

class Adversary:
	def __init__(self, N):
		self.N = self.N

	@property
	def loss(self):
		return np.random.uniform(0,1,size=[self.N])

class Player:
	def __init__(self, T, N):
		self.regret = 0
		self.T = T
		self.N = N
		self.t = 0
		self.I = np.zeros(T)
		self.max_regret = np.zeros(T)

	def play(self,observe,loss):
		I = self.choose_arm()
		N = np.size(observe(I))
		self.I[self.t] = I
		self.regret += loss[I] - loss
		self.max_regret[self.t]  = np.max(self.regret)
		t += 1

	def choose_arm(self):
		return np.random.randint(0,self.N)

class Game:
	def __init__(self, graph, adversary, T, players):
		self.graph = graph
		self.adversary = adversary
		self.T = T

	def round(self):
		observe = self.graph.observe
		loss = self.adversary.loss
		for player in self.players:
			player.play(observe,loss)

	def run(self):
		for i in range(0,self.T):
			self.round()

class DuplexpPlayer(Player):
	def __init__(self,T):
		super(T)
		Toff = self.T + 2
		Lhat = np.zeros([Toff,self.N])
		DLhat = np.zeros([Toff,self.N])
		p = np.zeros([Toff,self.N])
		O = np.zeros([Toff,self.N])
		estimate_phase = False

	def choose_arm(self):
		if self.estimate_phase:
			return super('choose_arm')
			#TODO perform estimation
		else:
			t = self.t
			e = t % 2
			eta = np.sqrt(np.log(self.N)/(self.N*self.N)
						  + np.sum(np.multiply(self.p[e:t:2],self.DLhat[e:t:2],self.DLhat[e:t:2])))
			w = 1/self.N * np.exp(-eta*self.Lhat[t-2,:])
			p = w/np.sum(self.w)
			I[t] = np.random.choice(N,p=p[t,:])

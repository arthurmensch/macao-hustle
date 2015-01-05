__author__ = 'arthur'

import numpy as np
import matplotlib.pyplot as plt

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
		return obs

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
		I = self.choose_arm(observe,loss)
		self.I[self.t] = I
		self.regret += loss[I] - loss
		self.max_regret[self.t]  = np.max(self.regret)
		self.t += 1

	def choose_arm(self,observe,loss):
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

	def display(self):
		for player in self.players:
			plt.plot(np.arange(player.T),player.max_regret)
		plt.show()

class DuplexpPlayer(Player):
	def __init__(self,T,graph):
		super(T)
		self.graph = graph
		Toff = self.T + 2
		self.Lhat = np.zeros([Toff,self.N])
		self.DLhat = np.zeros([Toff,self.N])
		self.p = np.zeros([Toff,self.N])
		self.O = np.zeros([Toff,self.N])
		self.estimate_phase = False
		self.last_t = np.zeros([2,np.graph.num_cluster])

	def choose_arm(self,observe,loss):
		if self.estimate_phase:
			return super('choose_arm')
			#TODO perform estimation
		else:
			K = np.zeros(self.N)
			G = np.zeros(self.N)
			t = self.t
			e = t % 2
			eta = np.sqrt(np.log(self.N)/(self.N*self.N)
						  + np.sum(np.multiply(self.p[e:t:2],self.DLhat[e:t:2],self.DLhat[e:t:2])))
			w = 1/self.N * np.exp(-eta*self.Lhat[t-2,:])
			I = np.random.choice(self.N,p=self.p[t,:])
			self.O[t,:] = observe(self.I[t])
			c = self.graph.find_cluster(self.I[t])
			M = np.zeros(self.graph.num_cluster)
			for k in range(0,self.graph.num_cluster):
				Oc = self.O[self.last_t[e,c],self.graph.C[k]]
				if k == c:
					Oc = np.delete(Oc,self.I[self.last_t[e][c]])
				min_i = np.nonzero(Oc)[0]
				M[k] = min_i[0] if np.size(min_i) > 0 else self.graph.cluster_size(k)-1
			for i in range(0,self.N):
				c= self.graph.find_cluster(i)
				K[i] = np.random.geometric(self.p[t,i])
				G[i] = min(M[c],K[i])
			lhat = np.multiply(loss,self.O[t,:],G)
			self.DLhat[t,:] = lhat
			self.Lhat[t,:] = self.Lhat[t-2,:]+self.DLhat[t,:]
			return I


def test():
	graph = Graph([500,500],np.array([[0.1,0.05],[0.05,0.2]]))
	adversary = Adversary(graph.N)
	players = [DuplexpPlayer(1000,)]

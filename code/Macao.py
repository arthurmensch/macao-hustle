__author__ = 'arthur'

import numpy as np
import matplotlib.pyplot as plt

class Graph:
	def __init__(self, cluster_size, r_mat):
		self.cluster_size = cluster_size
		self.num_cluster = np.size(self.cluster_size)
		self.cluster_bounds = np.zeros(self.num_cluster+1,dtype=int)
		self.cluster_bounds[1:self.num_cluster+1] = np.cumsum(cluster_size)
		self.N = self.cluster_bounds[self.num_cluster]
		self.nodes = np.arange(self.N)
		self.C = []
		for i in range(0,self.num_cluster):
			self.C.append(self.nodes[self.cluster_bounds[i]:self.cluster_bounds[i+1]])
		self.r_mat = r_mat

	def find_cluster(self, I):
		return np.searchsorted(self.cluster_bounds,I,side='right')-1

	def observe(self, I):
		c = self.find_cluster(I)
		obs = np.zeros(self.N)
		for i in range(0,self.num_cluster):
			obs[self.cluster_bounds[i]:self.cluster_bounds[i+1]] = np.random.choice(2, size=self.cluster_size[i], p =[1-self.r_mat[c,i], self.r_mat[c,i]])
		obs[I] = 1
		return obs

class Adversary:
	def __init__(self, T, N):
		self.N = N
		self.T = T
		N1 = int(self.N / 2)
		N2 = self.N - N1
		self.loss = np.hstack((np.random.uniform(5,10,size=[T,N1]),np.random.uniform(0,1,size=[T,N1])))

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
		self.regret += loss[self.t,I] - loss[self.t]
		self.max_regret[self.t]  = np.max(self.regret)
		self.t += 1

	def choose_arm(self,observe,loss):
		return np.random.randint(0,self.N)

	def __str__(self):
		return 'Naive'

class Game:
	def __init__(self, graph, adversary, T, players,number):
		self.graph = graph
		self.adversary = adversary
		self.T = T
		self.players = players
		self.number = number

	def round(self):
		observe = self.graph.observe
		loss = self.adversary.loss
		for player in self.players:
			player.play(observe,loss)

	def run(self):
		for i in range(0,self.T):
			self.round()

	def display(self):
		i = 1
		for player in self.players:
			plt.plot(np.arange(self.T),player.max_regret,label = str(player))
			i += 1
		plt.legend()
		plt.show()

class DuplexpPlayer(Player):
	def __init__(self,T,N,graph):
		Player.__init__(self,T,N)
		self.graph = graph
		Toff = self.T + 2
		self.Lhat = np.zeros([Toff,self.N])
		self.DLhat = np.zeros([Toff,self.N])
		self.p = np.zeros([Toff,self.N])
		self.O = np.zeros([Toff,self.N])
		self.estimate_phase = False
		self.last_t = np.zeros([2,self.graph.num_cluster])

	def choose_arm(self,observe,loss):
		if self.estimate_phase:
			return super('choose_arm')
			#TODO perform estimation
		else:
			K = np.zeros(self.N)
			G = np.zeros(self.N)
			t = self.t
			e = t % 2
			eta = np.sqrt(np.log(self.N)/((self.N*self.N)
						  + np.sum(np.multiply(np.multiply(self.p[e:t:2],self.DLhat[e:t:2]),self.DLhat[e:t:2]))))
			w = -eta*self.Lhat[t-2,:]
			w = w - np.max(w)
			w = np.exp(w)
			self.p[t,:] = w/np.sum(w)
			I = np.random.choice(int(self.N),p=self.p[t,:])
			self.O[t,:] = observe(I)
			c = self.graph.find_cluster(I)
			M = np.zeros(self.graph.num_cluster)
			for k in range(0,self.graph.num_cluster):
				Oc = self.O[self.last_t[e,c],self.graph.C[k]]
				if k == c:
					I_last = self.I[self.last_t[e,c]]
					Oc = np.delete(Oc,I_last-self.graph.cluster_bounds[self.graph.find_cluster(I_last)])
				min_i = np.nonzero(Oc)[0]
				M[k] = min_i[0] if np.size(min_i) > 0 else self.graph.cluster_size[k]-1
			self.last_t[e,c] = t
			G = np.zeros(self.N)
			for i in range(0,self.N):
				c= self.graph.find_cluster(i)
				K = np.random.geometric(self.p[t,i])
				G[i] = min(M[c],K)
			lhat = np.multiply(np.multiply(loss[t],self.O[t,:]),G)
			self.DLhat[t,:] = lhat
			self.Lhat[t,:] = self.Lhat[t-2,:]+self.DLhat[t,:]
			return I

	def __str__(self):
		return 'Duplexp'

class DuplexpPlayerErdos(Player):
	def __init__(self,T,N):
		Player.__init__(self,T,N)
		Toff = self.T + 2
		self.Lhat = np.zeros([Toff,self.N])
		self.DLhat = np.zeros([Toff,self.N])
		self.p = np.zeros([Toff,self.N])
		self.O = np.zeros([Toff,self.N])
		self.estimate_phase = False
		self.last_t = np.zeros(2)


	def choose_arm(self,observe,loss):
		if self.estimate_phase:
			return super('choose_arm')
			#TODO perform estimation
		else:
			K = np.zeros(self.N)
			G = np.zeros(self.N)
			t = self.t
			e = t % 2
			eta = np.sqrt(np.log(self.N)/((self.N*self.N)
						  + np.sum(np.multiply(np.multiply(self.p[e:t:2],self.DLhat[e:t:2]),self.DLhat[e:t:2]))))
			w = -eta*self.Lhat[t-2,:]
			w = w - np.max(w)
			w = np.exp(w)
			self.p[t,:] = w/np.sum(w)
			I = np.random.choice(int(self.N),p=self.p[t,:])
			self.O[t,:] = observe(I)
			Oc = self.O[self.last_t[e],:]
			I_last = self.I[self.last_t[e]]
			Oc = np.delete(Oc,I_last)
			min_i = np.nonzero(Oc)[0]
			M = min_i[0] if np.size(min_i) > 0 else self.N-1
			self.last_t[e] = t
			G = np.zeros(self.N)
			for i in range(0,self.N):
				K = np.random.geometric(self.p[t,i])
				G[i] = min(M,K)
			lhat = np.multiply(np.multiply(loss[t,:],self.O[t,:]),G)
			self.DLhat[t,:] = lhat
			self.Lhat[t,:] = self.Lhat[t-2,:]+self.DLhat[t,:]
			return I

	def __str__(self):
		return 'Duplexp Erdos'

def test():
	T = 1000
	#graph = Graph([200,200,200],np.array([[0.1,0.05,0],[0.05,0.2,0],[0,0,0.5]]))
	graph = Graph([600],np.array([[0.2]]))
	adversary = Adversary(T,graph.N)
	players = [DuplexpPlayer(T,graph.N,graph),DuplexpPlayerErdos(T,graph.N),Player(T,graph.N)]
	game = Game(graph,adversary,T,players,1)
	game.run()
	game.display()

if __name__ == '__main__':
	test()

__author__ = 'arthur'

import numpy as np

import matplotlib as mpl

mpl.use("pgf")
pgf_with_pdflatex = {
	"pgf.texsystem": "pdflatex",
	"pgf.preamble": [
		 r"\usepackage[utf8x]{inputenc}",
		 r"\usepackage[T1]{fontenc}",
		 r"\usepackage{amsmath}"
		 ]
}
mpl.rcParams.update(pgf_with_pdflatex)

import matplotlib.pyplot as plt
import pickle


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
	def __init__(self, game):
		self.game = game
		self.N = game.N
		self.T = game.T
		N1 = int(self.N / 2)
		N2 = self.N - N1
		self.loss = np.hstack((np.random.uniform(5,10,size=[self.T,N1]),np.random.uniform(0,1,size=[self.T,N1])))

class Game:
	def __init__(self, graph, T, players_type,number):
		self.graph = graph
		self.N = self.graph.N
		self.T = T
		self.number = number
		self.players_type = players_type
		self.num_players = len(self.players_type)
		self.max_regret = np.zeros([self.T,self.num_players])


	def init(self):
		self.adversary = Adversary(self)
		self.players =  [self.players_type[i](self) for i in range(0,self.num_players)]

	def round(self):
		observe = self.graph.observe
		loss = self.adversary.loss
		for player in self.players:
			player.play(observe,loss)

	def run(self):
		for i in range(0,self.number):
			self.init()
			for i in range(0,self.T):
				self.round()
			for j in range(0,self.num_players):
				self.max_regret[:,j] += self.players[j].max_regret
		self.max_regret /= self.number
		pickle.dump(self.max_regret,open('max_regret.p','wb+'))

	def display(self):
		self.max_regret = pickle.load(open('max_regret.p','rb'))
		for i in range(0, self.num_players):
			plt.plot(np.arange(self.T), self.max_regret[:, i], label=self.players_type[i].playerName)
		plt.legend(loc=2)
		plt.savefig('regret.pdf')


class Player:
	def __init__(self, game):
		self.regret = 0
		self.T = game.T
		self.N = game.N
		self.t = 0
		self.I = np.zeros(self.T)
		self.max_regret = np.zeros(self.T)

	def play(self,observe,loss):
		I = self.choose_arm(observe,loss[self.t,:])
		self.I[self.t] = I
		self.regret += loss[self.t,I] - loss[self.t]
		self.max_regret[self.t]  = np.max(self.regret)
		self.t += 1

	def choose_arm(self,observe,loss):
		return np.random.randint(0,self.N)

	playerName = 'Naive Player'

class DuplexpPlayer(Player):
	def __init__(self,game):
		Player.__init__(self,game)
		self.graph = game.graph
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
			lhat = np.multiply(np.multiply(loss,self.O[t,:]),G)
			self.DLhat[t,:] = lhat
			self.Lhat[t,:] = self.Lhat[t-2,:]+self.DLhat[t,:]
			return I

	playerName = 'Duplex SBM'


class DuplexpPlayerErdos(Player):
	def __init__(self,game):
		Player.__init__(self,game)
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
			lhat = np.multiply(np.multiply(loss,self.O[t,:]),G)
			self.DLhat[t,:] = lhat
			self.Lhat[t,:] = self.Lhat[t-2,:]+self.DLhat[t,:]
			return I

	def estimate_phase(self):
		k = np.int(np.ceil(math.e * np.log(self.T)/2))
		C=np.zeros(self.graph.num_cluster+1)
		for i in range(0,self.graph.num_cluster):
			C[i]= np.int(np.ceil(2*np.log(self.T)/self.graph.cluster_size[i]))
		j = 0
		s=0
		c = np.zeros(2*self.graph.num_cluster-1)
		isNul=np.zeros(2*self.graph.num_cluster-1)
		r=np.zeros(2*self.graph.num_cluster-1)
		regret = 0
		max_regret = []
		for i in range(0,self.graph.num_cluster):
			for t in range(s,min(max(s+C[i],s+C[i+1],T))):
				l = loss()
				I = np.random.randint(self.graph.cluster_bounds[i],self.graph.cluster_bounds[i+1])
				regret += l[I]-l
				max_regret.append(np.max(regret))
				O, O_index = observed(I)
				if (t-s)<C[i] :
					c[i] += np.sum(O[self.graph.cluster_bounds[i]:self.graph.cluster_bounds[i+1]]) - 1
				if i<self.graph.num_cluster-1 and (t-s)<C[i+1]:
					c[self.graph.num_cluster+i] +=np.sum(0[self.graph.cluster_bounds[i+1]:self.graph.cluster_bounds[i+2]])
			s=t
			if c[i]/(C[i]*(self.graph.cluster_size[i]-1))<=3/2*self.graph.cluster_size[i]:
				isNul[i]=1;
			if i<self.graph.num_cluster-1 and c[self.graph.num_cluster+i]/(C[i+1]*self.graph.cluster_size[i+1])<=3/(2*self.graph.cluster_size[i+1]):
				isNull[self.graph.num_cluster+i]=1;
			

		for i in range(0,2*self.graph.num_cluster-1):
			if isNull[i]==0:
				break
		if i==2*self.graph.num_cluster-2:
			self.t=s
			return r
		else:
			for i in range(0,self.graph.num_cluster):
				for t in range(s,T):
					l = loss()
					I= np.random.randint(self.graph.cluster_bounds[i],self.graph.cluster_bounds[i+1])
					regret += l[I]-l
					max_regret.append(np.max(regret))
					O, O_index = observed(I)
					M1 = np.zeros(k)
					M2 = np.zeros(k)
					if r[i]==0:
						for i in range(self.graph.cluster_bounds[i],self.graph.cluster_bounds[i+1]):
							M1[j1] = M1[j1] + (i != I)
							j1 = j1 + O[i] * (i != I)
							if j1 == k:
								r[i]=1/(np.max(M1)+1)

					if i<self.graph.num_cluster-1:
						if r[self.graph.num_cluster+i]==0:
							for i in range(self.graph.cluster_bounds[i+1],self.graph.cluster_bounds[i+2]):
								M2[j2] = M2[j2] + (i != I)
								j2 = j2 + O[i] * (i != I)
								if j2 == k:
									r[self.graph.num_cluster+i]=1/(np.max(M2)+1)

				if r[i]>0 and( i==self.graph.num_cluster-1 or r[self.graph.num_cluster+i]>0 ) :
					s=t
					break
			self.t=t
			self.regret=np.array(max_regret)
			return r

	playerName = 'Duplex Erdös'

def test():
	T = 10
	num = 50
	graph = Graph([200,200,200],np.array([[0.1,0.05,0],[0.05,0.2,0],[0,0,0.5]]))
	#graph = Graph([600],np.array([[0.2]]))
	players_type = [DuplexpPlayer,DuplexpPlayerErdos,Player]
	game = Game(graph,T,players_type,num)
	game.run()
	game.display()

if __name__ == '__main__':
	test()

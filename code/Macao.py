__author__ = 'arthur'

import numpy as np
import pickle
import math
import matplotlib as mpl
import graph_tool.all as gt

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

	def draw(self):
		g, bm = gt.random_graph(self.N, lambda i,b: np.random.poisson(sum([self.cluster_size[c]*(self.r_mat[b,c]) for c in range(0,self.num_cluster)])/10),
							directed=False, model='blockmodel-traditional',
							block_membership=lambda i: self.find_cluster(i),vertex_corr=lambda i,j: self.r_mat[i,j])
		gt.graph_draw(g, vertex_fill_color=bm, edge_color="black", output="blockmodel.png")


class Adversary:
	def __init__(self, game):
		self.game = game
		self.N = game.N
		self.T = game.T
		self.loss = np.zeros([self.T,self.N])
		bias = np.arange(self.game.graph.num_cluster)*10
		for i in range(0,self.game.graph.num_cluster):
			self.loss[:,self.game.graph.C[i]] = bias[i]#+ np.tile(np.random.normal(0,5,size=[1,self.game.graph.cluster_size[i]]), reps=[self.T,1])
		self.loss += np.random.normal(0,1,size=[self.T,self.N])
		#self.loss = np.random.uniform(0,1,size=[self.T,self.N])
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
		#pickle.dump(self.max_regret, open('data/max_regret.p','wb+'))

	def display(self,j=0):
		#		pickle.load(open('data/max_regret.p','rb'))
		for i in range(0,self.num_players):
			plt.plot(np.arange(self.T),self.max_regret[:,i],label = self.players_type[i].playerName)
		plt.legend(loc=2)
		plt.xlabel('Time')
		plt.ylabel('Maximal expected regret')
		plt.savefig('data/regret'+str(j)+'.pdf')
		plt.close()


class Player:

	playerName = 'Naive Player'
	def __init__(self, game):
		self.regret = 0
		self.T = game.T
		self.N = game.N
		self.t = 0
		self.I = np.zeros(self.T)
		self.max_regret = np.zeros(self.T)

	def play(self,observe,loss):
		if self.t < self.T:
			I = self.choose_arm(observe,loss[self.t,:])
			self.I[self.t] = I
			self.regret += loss[self.t,I] - loss[self.t,:]
			self.max_regret[self.t]  = np.max(self.regret)
			self.t += 1

	def choose_arm(self,observe,loss):
		return np.random.randint(0,self.N)

class DuplexpPlayer(Player):
	playerName = 'Duplexp SBM'

	def __init__(self,game):
		Player.__init__(self,game)
		self.graph = game.graph
		self.A = 1
		self.num_episode = int(np.ceil(self.T/self.A))
		Toff = self.T + 2
		episodeOff = self.num_episode + 2
		self.Lhat = np.zeros([episodeOff,self.N])
		self.DLhat = np.zeros([episodeOff,self.N])
		self.p = np.zeros([Toff,self.N])
		self.O = np.zeros([Toff,self.N])
		self.estimate_phase = True
		self.last_t = np.zeros([2,self.graph.num_cluster])
		self.last_t_immediate = self.last_t.copy()
		self.episode  = 1 #offset
		self.tstart = self.t
		self.rbar = np.zeros([self.graph.num_cluster,self.graph.num_cluster])

	def play(self,observe,loss):
		if self.estimate_phase :
			rbar,I,s=self.estimate_r(observe,loss)
			for k in range(0,s):
				self.I[self.t+k]=I[k]
				self.regret += loss[self.t+k,I[k]] - loss[self.t+k,:]
				self.max_regret[self.t+k] = np.max(self.regret)
			self.t+=s
			self.tstart = self.t
			self.rbar = rbar
			#print(rbar)
			rstar = min([self.rbar[i,j]*self.graph.cluster_size[j] if self.rbar[i,j] > 0 else self.N for i,j in zip(range(0,self.graph.num_cluster),range(0,self.graph.num_cluster))])
			self.A = max(1,int(np.ceil(np.log(self.T))/rstar))
			#self.A =4
			print(self.A)
			self.estimate_phase=False
		else :
			super(DuplexpPlayer,self).play(observe,loss)
			
	def choose_arm(self,observe,loss):
		t = self.t
		if (self.t - self.tstart) % self.A == 0:
			self.episode += 1
			e = self.episode % 2
			eta = np.sqrt(np.log(self.N)/((self.N*self.N)
						  + np.sum(np.multiply(np.multiply(self.p[e:self.episode:2,:],self.DLhat[e:self.episode:2,:]),self.DLhat[e:self.episode:2,:]))))
			w = -eta*self.Lhat[self.episode-2,:]
			w = w - np.max(w)
			#print(w[0:100])
			w = np.exp(w)
			if np.sum(w) == 0:
				raise ValueError
			self.p[t,:] = w/np.sum(w)
			if any(np.isnan(self.p[t,:])):
				raise ValueError
			self.last_t[e,:] = self.last_t_immediate[e,:].copy()
		else: #no p update
			e = self.episode % 2
			self.p[t,:] = self.p[t-1,:]
		I = np.random.choice(int(self.N),p=self.p[t,:])
		self.O[t,:] = observe(I)
		cI = self.graph.find_cluster(I)
		M = np.zeros(self.graph.num_cluster)
		for k in range(0,self.graph.num_cluster):
			#print(k)
			Oc = self.O[self.last_t[e,cI],self.graph.C[k]]
			if k == cI:
				I_last = self.I[self.last_t[e,cI]] if self.last_t[e,cI] >= 2 else 0
				cI_last = self.graph.find_cluster(I_last)
				Oc = np.delete(Oc,I_last-self.graph.cluster_bounds[cI_last]) #max for first 2 phase (hack)
			min_i = np.nonzero(Oc)[0]
			M[k] = min_i[0] if np.size(min_i) > 0 else self.graph.cluster_size[k]-1
		self.last_t_immediate[e,cI] = t
		G = np.zeros(self.N)
		for i in range(0,self.N):
			ci= self.graph.find_cluster(i)
			if self.rbar[cI,ci] != 0:
				K = max(0,np.random.geometric(self.p[t,i]))
				G[i] = min(M[ci],K)
			else:
				G[i] = 1 / self.p[t,I]
		lhat = np.multiply(np.multiply(loss,self.O[t,:]),G)
		self.DLhat[self.episode,:] += lhat
		if (self.t + 1 - self.tstart) % self.A == 0:
			self.Lhat[self.episode,:] = self.Lhat[self.episode-2,:]+self.DLhat[self.episode,:]
		return I

	def estimate_r(self,observe,loss):
		T=self.T
		I=[]
		s_int=0
		k = np.int(np.ceil(math.e * np.log(self.T)/2))
		C=np.zeros(self.graph.num_cluster)
		for i in range(0,self.graph.num_cluster):
			C[i]= np.int(np.ceil(2*np.log(self.T)/self.graph.cluster_size[i]))
		s=0
		c = np.zeros((self.graph.num_cluster,self.graph.num_cluster))
		is_nul=np.zeros((self.graph.num_cluster,self.graph.num_cluster))
		r=np.zeros((self.graph.num_cluster,self.graph.num_cluster))
		for i in range(0,self.graph.num_cluster):
			lim=int(min(s+np.max(C),T))
			for t in range(s,lim):
				l = loss[t]
				I.append(np.random.randint(self.graph.cluster_bounds[i],self.graph.cluster_bounds[i+1]))
				O = observe(I[t])
				self.O[t,:]=O
				for j in range(0,self.graph.num_cluster):
					if (t-s)<C[j] and i==j:
						c[i,j] += np.sum(O[self.graph.cluster_bounds[j]:self.graph.cluster_bounds[j+1]]) - 1
					elif (t-s)<C[j]:
						c[i,j] +=np.sum(O[self.graph.cluster_bounds[j]:self.graph.cluster_bounds[j+1]])
				s_int=t+1
			s=s_int
			for j in range(0,self.graph.num_cluster):
				if i==j and c[i,i]/(C[i]*(self.graph.cluster_size[i]-1))<=3/(2*self.graph.cluster_size[i]):
					is_nul[i,i]=1;
				if c[i,j]/(C[j]*self.graph.cluster_size[j])<=3/(2*self.graph.cluster_size[j]):
					is_nul[i,j]=1;
		all_nul=1
		for i in range(0,self.graph.num_cluster):
			for j in range(0,self.graph.num_cluster):
				if is_nul[i,j]==0:
					all_nul = 0
					break
			if all_nul==0:
				break
		if all_nul==1:
			#all rbar are nul, equal to 0
			return r,I,s

		else:
			for i in range(0,self.graph.num_cluster):
				all_nul=1
				for j in range(0,self.graph.num_cluster):
					if(is_nul[i,j]==0):
						all_nul=0
						break

				if all_nul==1:
					continue
				M = np.zeros((self.graph.num_cluster,k))
				ind=np.zeros(self.graph.num_cluster)
				for t in range(s,T):
					l = loss[t]
					I.append(np.random.randint(self.graph.cluster_bounds[i],self.graph.cluster_bounds[i+1]))
					O= observe(I[t])
					self.O[t,:]=O
					s_int=t+1
					for j in range(0,self.graph.num_cluster):
						if r[i,j]==0 and is_nul[i,j]==0:
							for m in range(self.graph.cluster_bounds[j],self.graph.cluster_bounds[j+1]):
								M[j,ind[j]] = M[j,ind[j]] + (m != I[t])
								ind[j] = ind[j] + O[m] * (m != I[t])
								if ind[j] == k:
									r[i,j]=1/(np.max(M[j,:])+1)
									break
								elif m == self.graph.cluster_bounds[j+1]-1:
									M[j,ind[j]]=0
					next_class=0
					for j in range(0,self.graph.num_cluster):
						next_class+=(r[i,j]>0 or is_nul[i,j]==1)
					if next_class==self.graph.num_cluster:
						s=s_int
						break
				s=s_int
			return r,I,s

class DuplexpPlayerErdos(DuplexpPlayer):
	playerName = 'Duplexp Erdös'

	def __init__(self,game):
		super(DuplexpPlayerErdos,self).__init__(game)
		self.graph = Graph([sum(self.graph.cluster_size)],[1])


class ExpPlayer(DuplexpPlayer):
	playerName = 'Exp player'

	def __init__(self,game):
		super(ExpPlayer,self).__init__(game)

	def play(self,observe,loss):
		def observe_exp(I):
			A = np.zeros(self.N)
			A[I] = 1
			return A
		if self.t < self.T:
			I = self.choose_arm(observe_exp,loss[self.t,:])
			self.I[self.t] = I
			self.regret += loss[self.t,I] - loss[self.t,:]
			self.max_regret[self.t]  = np.max(self.regret)
			self.t += 1
def test():
	for i in range(0,1):
		T = 500
		num = 1
		#graph = Graph([50,10,50,10],np.array([[0.1,0.2,0.7,0.1],[0.6,0.1,0.2,0.9],[0.1,0.2,0.8,0.4],[0.4,0.1,0.9,0.5]]))
		graph = Graph([100,1000],np.array([[0.9,0.01],[0.01,0.1]]))
		#graph = Graph([600],np.array([[0.2]]))
		#graph = Graph([100,100,100,100],associative_array(4))
		graph.draw()
		players_type = [DuplexpPlayer,DuplexpPlayerErdos,ExpPlayer]
		game = Game(graph,T,players_type,num)
		game.run()
		game.display(i)


def associative_array(N):
	res = np.ones([N,N]) * 0.001
	connect = [0.1, 0.1, 0.1, 0.7]
	for i in range(0,N):
		res[i,i] = connect[i]
	return res/10

def neighbour_array(N):
	res = np.ones([N,N]) * 0.001
	for i in range(0,N):
		res[i,i] = 0.5
		if i > 0:
			res[i,(i-1) % N] = 0.1
		if i < N-1:
			res[i,(i+1) % N] = 0.1
	return res


def draw_graph():
	#graph = Graph([100,1000],np.array([[0.9,0.02],[0.02,0.1]])/10)
	#graph = Graph([500,100,500,100],np.array([[0.1,0.2,0.7,0.1],[0.6,0.1,0.2,0.9],[0.1,0.2,0.8,0.4],[0.4,0.1,0.9,0.5]]))
	graph.draw()

if __name__ == '__main__':
	test()

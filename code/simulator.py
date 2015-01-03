__author__ = 'arthur'

import numpy as np
import math
import matplotlib.pyplot as plt

N = 1000
T = 1000
r = 0.01

def loss():
	return np.random.uniform(0,1,size=[N])
	#return np.arange(N)

def observed(I):
	rand = np.random.choice(2, size=N, p =[1-r, r])
	rand[I] = 1
	return rand, np.nonzero(rand)[0]

def play_naive():
	regret = np.zeros(N)
	max_regret = np.zeros(T)
	for t in range(0,T):
		l = loss()
		I = np.random.randint(0,N)
		obs = observed(I)
		observed_loss = np.hstack((obs, l[obs]))
		regret += l[I]-l
		max_regret[t] = np.max(regret)
	plt.plot(np.arange(T),max_regret)
	plt.show()

def play_duplex():
	rbar,tstart,max_regret_start = estimate_r()
	regret = max_regret_start[tstart-2]
	Toff = T+2-tstart

	thres = np.log(Toff)/(2*N)

	if rbar == 0:
		return exp3()
	elif rbar < thres:
		A = np.int(np.ceil(np.log(Toff)/(N*rbar)))
	else:
		A = 1


	Lhat = np.zeros([Toff,N])
	DLhat = np.zeros([Toff,N])
	O = np.zeros([Toff,N])
	eta = np.zeros([Toff,N])
	p = np.zeros([Toff,N])
	l = np.zeros([Toff,N])
	lhat = np.zeros([Toff,N])
	w = np.zeros([Toff,N])
	W = np.zeros(Toff)
	I = np.zeros([Toff])
	observed_loss = []

	M = np.zeros([Toff])
	K = np.zeros([Toff,N])
	G = np.zeros([Toff,N])

	regret = np.zeros(N)
	regret_naive = np.zeros(N)
	max_regret = np.zeros(Toff)
	max_regret_naive = np.zeros(Toff)
	I_naive = np.zeros(Toff)

	#round = np.int(Toff / A)
	for t in range(2,Toff+1-A,A):
		e = t % 2
		eta[t] = np.sqrt(np.log(N)/(A*A*N*N) + np.sum(np.multiply(p[e:t:2],DLhat[e:t:2],DLhat[e:t:2])))
		w[t,:] = 1/N * np.exp(-eta[t]*Lhat[t-2,:])
		W[t] = np.sum(w[t,:])
		p[t,:] = w[t,:]/ W[t]

		for j in range(0,A):
			l[t+j,:] = loss()
			I[t+j] = np.random.choice(N,p=p[t,:])
			O[t+j,:], O_index = observed(I[t+j])
			observed_loss.append(np.hstack((O_index, l[t+j,O_index])))
			Od = np.nonzero(np.delete(O[t+j,:],I[t+j]))
			if np.size(Od[0]) >0:
				M[t+j] = Od[0][0]
			else:
				M[t+j] = N-1
			for i in range(0,N):
				K[t+j,i] = np.random.geometric(p[t,i])
				G[t+j,i] = min(M[t+j],K[t+j,i])
			lhat[t+j,:] = np.multiply(l[t+j,:],O[t+j,:],G[t+j,:])
			regret += l[t+j,I[t+j]]-l[t+j,:]
			max_regret[t+j] = np.max(regret)
			I_naive[t+j] = np.random.randint(0,1000)
			regret_naive += l[t+j,I_naive[t+j]]-l[t+j,:]
			max_regret_naive[t+j] = np.max(regret_naive)
		DLhat[t,:] = np.sum(lhat[t:t+A,:],axis=0)
		Lhat[t,:] = Lhat[t-2,:]+DLhat[t,:]

	max_regret = np.hstack((max_regret_start,max_regret[2:]))
	plt.plot(np.arange(np.size(max_regret)),max_regret)
	plt.show()

def estimate_r():
	k = np.int(np.ceil(math.e * np.log(T)/2))
	C = np.int(np.ceil(2*np.log(T)/N))
	j = 0
	c = 0
	regret = 0
	max_regret = []
	for t in range(0,C):
		l = loss()
		I = np.random.randint(0,N)
		regret += l[I]-l
		max_regret.append(np.max(regret))
		O, O_index = observed(I)
		c += np.sum(O) - 1
	if c/(C*(N-1)) <= 3 /(2*N):
		return 0,C,np.array(max_regret)
	else:
		for t in range(C,T):
			l = loss()
			I= np.random.randint(0,N)
			regret += l[I]-l
			max_regret.append(np.max(regret))
			O, O_index = observed(I)
			M = np.zeros(k)
			for i in range(0,N):
				M[j] = M[j] + (i != I)
				j = j + O[i] * (i != I)
				if j == k:
					return 1/(np.max(M)+1),t+1,np.array(max_regret)
		return 0,T,np.array(max_regret)
	#TODO

if __name__ == '__main__':
	play_duplex()
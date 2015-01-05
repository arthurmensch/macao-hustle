__author__ = 'arthur'

import numpy as np
import math
import matplotlib.pyplot as plt

N1 = 1000
N2 = 1000
N = N1 + N2
C1 = np.arange(N1)
C2 = np.arange(N2)+N1
T = 1000
r11 = 0.1
r22 = 0.2
r12 = 0.05
r21 = 0.05

def loss():
	return np.hstack((np.random.uniform(4,8,size=[200]),np.random.uniform(0,1,size=[1800])))
	#return np.arange(N)

def observed(I):
	rand = np.zeros(N)
	if I in C1:
		rand[C1] = np.random.choice(2, size=N1, p =[1-r11, r11])
		rand[I] = 1
		rand[C2] = np.random.choice(2, size=N2, p =[1-r12, r12])
	else:
		rand[C2] = np.random.choice(2, size=N2, p =[1-r22, r22])
		rand[I] = 1
		rand[C1] = np.random.choice(2, size=N1, p =[1-r21, r21])

	return rand#, np.nonzero(rand[C1])[0],np.nonzero(rand[C2])[0]

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
	thres = np.log(T)/(2*N)

	if rbar == 0:
		return exp3()
	elif rbar < thres:
		A = np.int(np.ceil(np.log(T)/(N*rbar)))
	else:
		A = 1

	Toff = T+1+A-tstart



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
	for t in range(1+A,Toff+1-A,A):
		e = t % 2
		eta[t] = np.sqrt(np.log(N)/(A*A*N*N) + np.sum(np.multiply(p[e:t:2],DLhat[e:t:2],DLhat[e:t:2])))
		w[t,:] = 1/N * np.exp(-eta[t]*Lhat[t-2,:])
		W[t] = np.sum(w[t,:])
		p[t,:] = w[t,:]/ W[t]

		for j in range(0,A):
			l[t+j,:] = loss()
			I[t+j] = np.random.choice(N,p=p[t,:])
			O[t+j,:] = observed(I[t+j])
			#observed_loss.append(np.hstack((O_index, l[t+j,O_index])))
			Od = np.nonzero(np.delete(O[t-A+j,:],I[t-A+j]))
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

	max_regret = np.hstack((max_regret_start,max_regret[1+A:]))
	return max_regret
	#plt.plot(np.arange(np.size(max_regret)),max_regret,np.arange(np.size(max_regret)),max_regret_naive)
	#plt.show()

def play_duplex_simple():
	Toff = T+2

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

	M = np.zeros([Toff,2])
	K = np.zeros([Toff,N])
	G = np.zeros([Toff,N])

	regret = np.zeros(N)
	regret_naive = np.zeros(N)
	max_regret = np.zeros(Toff)
	max_regret_naive = np.zeros(Toff)
	I_naive = np.zeros(Toff)

	last_t  = np.array([[0,0],[0,0]])

	#round = np.int(Toff / A)
	for t in range(2,Toff):
		e = t % 2
		eta[t] = np.sqrt(np.log(N)/(N*N) + np.sum(np.multiply(p[e:t:2],DLhat[e:t:2],DLhat[e:t:2])))
		w[t,:] = 1/N * np.exp(-eta[t]*Lhat[t-2,:])
		W[t] = np.sum(w[t,:])
		p[t,:] = w[t,:]/ W[t]

		l[t,:] = loss()
		I[t] = np.random.choice(N,p=p[t,:])
		cluster = 0 if I[t] in C1 else 1
		O[t,:] = observed(I[t])
		#observed_loss.append(np.hstack((O_index, l[t,O_index])))

		Od = [None] * 2
		Od[0] = np.nonzero(np.delete(O[last_t[e][cluster],C1],I[last_t[e][cluster]]) if cluster == 0 else O[last_t[e][cluster],C1])[0]
		if np.size(Od[0]) >0:
			M[t,0] = Od[0][0]
		else:
			M[t,0] = N1-1
		Od[1] = np.nonzero(np.delete(O[last_t[e][cluster],C2],I[last_t[e][cluster]]) if cluster == 1 else O[last_t[e][cluster],C2])[0]
		if np.size(Od[1]) >0:
			M[t,1] = Od[1][0]
		else:
			M[t,1] = N2-1

		for i in range(0,N):
			clu = 0 if i in C1 else 1
			K[t,i] = np.random.geometric(p[t,i])
			G[t,i] = min(M[t,clu],K[t,i])

		lhat[t,:] = np.multiply(l[t,:],O[t,:],G[t,:])
		regret += l[t,I[t]]-l[t,:]
		max_regret[t] = np.max(regret)
		I_naive[t] = np.random.randint(0,1000)
		regret_naive += l[t,I_naive[t]]-l[t,:]
		max_regret_naive[t] = np.max(regret_naive)
		DLhat[t,:] = lhat[t,:]
		Lhat[t,:] = Lhat[t-2,:]+DLhat[t,:]
		last_t[e][cluster] = t
		I[t] = np.nonzero(C1 == I[t])[0][0] if cluster == 0 else np.nonzero(C2 == I[t])[0][0]

	return max_regret
	# plt.plot(np.arange(np.size(max_regret)),max_regret)
	# plt.show()



def estimate_r_simple():
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

def estimate_r():
        k = np.int(np.ceil(math.e * np.log(T)/2))
        C = np.int(np.ceil(2*np.log(T)/N))
        B1= np.int(np.ceil(2*np.log(T)/N1))
        B2= np.int(np.ceil(2*np.log(T)/N2))
        j = 0
        c = 0
        regret = 0
        max_regret = []
	#first estimate r11 an r12
        for t in range(0,max(C1,C2)):
                l = loss()
                I = np.random.randint(0,N1)
                regret += l[I]-l
                max_regret.append(np.max(regret))
                O, O_index = observed(I)
                c1 += np.sum(O[C1]) - 1
                c12 +=np.sum(0[C2])

        #first extimate r22
        for t in range(max(C1,C2),C2+max(C1,C2)):
                l = loss()
                I = np.random.randint(0,N1)
                regret += l[I]-l
                max_regret.append(np.max(regret))
                O, O_index = observed(I)
                c2 += np.sum(O[C2]) - 1

        if c2/(B2*(N2-1)) <= 3 /(2*N2) and c1/(B1*(N1-1)) <= 3 /(2*N1) and c12/(B2*(N2)) <= 3 /(2*N2) :
                return 0,C1+C2,np.array(max_regret)

        else:
                r11=0
                r22=0
                r12=0
                hasr12=False
                hasr11=False
                #boucle pour r11 et r12
                for t in range(max(C1,C2)+C2,T):
                        l = loss()
                        I= np.random.randint(0,N1)
                        regret += l[I]-l
                        max_regret.append(np.max(regret))
                        O, O_index = observed(I)
                        M = np.zeros(k)
                        if not hasr11:
                                for i in range(0,N1):
                                        M[j] = M[j] + (i != I)
                                        j = j + O[i] * (i != I)
                                        if j == k:
                                                r11=max(1/(np.max(M)+1),r11)
                                                hasr11=True
                        if not hasr12:
                                for i in range(N1,N):
                                        M[j] = M[j] + (i != I)
                                        j = j + O[i] * (i != I)
                                        if j == k:
                                                r12=max(1/(np.max(M)+1),r12)
                                                hasr12=True
                        if hasr11 and hasr12:
                                break
                        
                #boucle pour r22
                for s in range(t,T):
                        l = loss()
                        I= np.random.randint(N1,N1+N2)
                        regret += l[I]-l
                        max_regret.append(np.max(regret))
                        O, O_index = observed(I)
                        M = np.zeros(k)
                        for i in range(N1,N1+N2):
                                M[j] = M[j] + (i != I)
                                j = j + O[i] * (i != I)
                                if j == k:
                                        r22=max(1/(np.max(M)+1),r22)
                                        break;
                return r11,r12,r22,s,np.array(max_regret)

if __name__ == '__main__':
	max_regret_r = play_duplex()
	max_regret_r1r2 = play_duplex_simple()
	plt.plot(np.arange(np.size(max_regret_r)),max_regret_r,np.arange(np.size(max_regret_r1r2)),max_regret_r1r2)
	plt.show()

macao-hustle
============

Reinforcement Learning Project

##Objectives

###Mail

Hi Michaël and Arthur,

I'm resending the (submitted) paper on random graph bandits with stochastic observation.  This paper treats a Facebook-like setting when the advertiser proposes something to a user and then gets the information ("like") not only from that user but also from his "friends".  Obviously, not from all the friends and not from all the nodes in the graph. The model that it is assumed is that every other node in the graph reveals its information with some (unknown) revelation probability "r". The paper then proposes an algorithm with an analysis for this setting.

In the variation of the project - a generalization, we say that in real social networks, it is not so simple as that every node reveals its information with the same probability.  We can rather see a social graph as a mixture of several communities (either from our perspective:family, work, friends -  or in general: communities of people by their countries, professions, etc.) ... Then each community would have a different revelation probability. 

Your tasks can be:

1) define a precise model 
2) propose an algorithm
3) perform experiments
4) attempt to analyze it

With respect to 1) your model can be simple enough in order to have an algorithm for it. The simplest possible could be just to consider 2 different cluster with different revelation probability r1 and r2. Or you can take a look at the "stochastic block model", which generates community-like graphs instead of Erdős-Rényi model currently considered.

Cheers,

Michal

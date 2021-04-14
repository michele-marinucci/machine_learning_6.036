"""Reinforcement Learning """

import numpy as np


#Q update
def update(self, data, lr):
    for s,a,t in data:
        q_update=self.get(s, a)+lr*(t-self.get(s, a))
        self.set(s,a,q_update)

#q learn
def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5, interactive_fn=None):
    s=mdp.init_state()
    i=0
    while i <iters:
        gamma= 0 if mdp.terminal(s) else mdp.discount_factor
        a=epsilon_greedy(q, s, eps)
        r, s_prime=mdp.sim_transition(s,a)
        t=r+gamma*value(q,s_prime)
        q.update([(s,a,t)],lr)
        s=s_prime
        i+=1
    return q
   
#batch Q-learn     
def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,episode_length=10, n_episodes=2,interactive_fn=None):
    
    
    all_experiences = []
    for i in range(iters):
        if interactive_fn: interactive_fn(q, i)
        
        """data generation"""
        for j in range(n_episodes):
            explore = lambda s: epsilon_greedy(q,s,eps)
            _, episode,_=sim_episode(mdp, episode_length, explore, False)
            all_experiences+=episode
    
        ''' update Q values - compute fresh targets for EVERY experience tuple'''
        all_q_targets = []
        for s,a,r,s_prime in all_experiences:
            temp= 0 if mdp.terminal(s) else mdp.discount_factor*value(q,s_prime)
            t=r+temp
            all_q_targets.append((s,a,t))
        
        #for i in range(iters):
            # include this line in the iteration, where i is the iteration number
            #if interactive_fn: interactive_fn(q, i)
        
        q.update(all_q_targets, lr)
        
    return q

































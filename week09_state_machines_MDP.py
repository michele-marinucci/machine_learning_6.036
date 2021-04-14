"""State Machines and Markov Decision Processes"""

import numpy as np
#State Machines

#transduce
class SM:
    start_state = None

    def transduce(self, input_seq):
        '''input_seq: a list of inputs to feed into SM
           returns:   a list of outputs of SM'''
    
        new=[]
        j=0
        for i in input_seq:
            j+=i
            new.append(j)
        return new
	
#binary addition
		
class Binary_Addition(SM):
    start_state = (0,0)


    def transition_fn(self, s, x):
        esum=x[0]+x[1]+s[1]
        if esum<=1:
            return (esum%2,0)
        else:
            return (esum%2,1)
    def output_fn(self, s):
        return s[0]
	
#reverser

class Reverser(SM):
    start_state = ([],0)

    def transition_fn(self, s, x):
        if s[1]==1:
            if len(s[0])!=0:
                return (s[0][1:],1)
            else:
                return (s[0],1) #doubt
        elif x=="end":
            return (s[0],1)
        else:
            return ([x]+s[0],s[1])     

    def output_fn(self, s):
        if len(s[0])==0:
            return None
        elif s[1]==0:
            return None
        elif s[1]==1:
            return s[0][0]

#Markov Decision Processes

def value(q, s):
    """ Return Q*(s,a) based on current Q

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q_star = value(q,0)
    >>> q_star
    10
    """
    #return max([q,get(s,i) for i in q.actions])
    return max(q.get(s,"c"),q.get(s,"b"))

def greedy(q, s):
    """ Return pi*(s) based on a greedy strategy.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> greedy(q, 0)
    'c'
    >>> greedy(q, 1)
    'b'
    """
    val = max(q.q[(s,a)] for a in q.actions)
    for i in q.q.items():
        if i[1]==val:
          return i[0][1]  

def epsilon_greedy(q, s, eps = 0.5):
    """ Returns an action.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> eps = 0.
    >>> epsilon_greedy(q, 0, eps) #greedy
    'c'
    >>> epsilon_greedy(q, 1, eps) #greedy
    'b'
    """
    if random.random() < eps:  # True with prob eps, random action
        return uniform_dist(q.actions).draw()
    else:
        return greedy (q,s)

#Q-value iteration

def value(q, s):
    return max(q.get(s, a) for a in q.actions)

def value_iteration(mdp, q, eps = 0.01, max_iters=1000):
    print(len(q.actions)*len(q.states))
    while True:
        new_q = q.copy()
        count = 0
        for s in q.states:
            for a in q.actions:
                
                reward = mdp.reward_fn(s,a)
                def func(i):
                    return value(q,i)
                term2 = mdp.transition_model(s,a).expectation(func)
                new_q.set(s,a,reward + mdp.discount_factor*term2)
                
                if np.abs(new_q.get(s,a) - q.get(s,a)) < eps:
                    count += 1
                   
        if count == len(q.actions)*len(q.states):
            return new_q
        q = new_q

#finite horizions
def q_em(mdp, s, a, h):
    if h==0:
        return 0
    else:
        # transition_model: function from (state, action) into DDist over next state
        
        p=mdp.transition_model(s,a).getAllProbs()
        esum=0
        for state, prob in p:
            best_combo=-1e100
            for action in mdp.actions:
                best_combo=max(q_em(mdp, state, action, h-1),best_combo)
            esum+=best_combo*prob
        return mdp.reward_fn(s,a)+mdp.discount_factor*esum
import numpy as np

num_observations = 4 #1, -2, 2, 0 reward levels
num_actions = 2 #cooperate, cheat
num_states = 4  #the possible combinations
#(cooperate & cooperate): ++, (cooperate & cheat): +-, (cheat&cooperate): -+ , (cheat&cheat): --

A = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

C = np.array([1, -2, 2, 0])



num_actions = 2 #cooperate, cheat 
""" Note that we don't really need to 'perform' actions because
the actions are embedded directly in the observations and states"""

num_states = 4 #the possible combinations 
#(cooperate & cooperate): ++, (cooperate & cheat): +-, (cheat&cooperate): -+ , (cheat&cheat): --

reward = [1, -2, 2, 0] #(cooperate & cooperate), (cooperate & cheat), (cheat&cooperate), (cheat&cheat)


""" In tit for tat, the player will cooperate until the opponent cheats 
Then respond to cheating by cheating, until the opponent cooperates again
and then continue cooperating

If this strategy is achieved in a noiseless system, then the players will converge to continuous cooperation p(s = ++)"""

B_tit_for_tat = np.array([[1,0,0,0],
                          [0,0,1,0],
                          [0,1,0,0],
                          [0,0,0,1]])


#p(++ | ++ ) = 1
#p( ++ | +- ) = 0
#p( ++ | -+ ) = 0
#p( ++ | -- ) = 0

#p( +- | ++ ) = 0
#p( +- | +-) = 0
#p( +- | -+ ) = 1
#p( +- | -- ) = 0

#p( -+ | ++ ) = 0
#p( -+ | +- ) = 1
#p( -+ | -+ ) = 0
#p( -+ | -- ) = 0

#p( -- | ++ ) = 0
#p( -- | +- ) = 0
#p( -- | -+ ) = 0
#p( -- | -- ) = 1


""" An issue occurs though if there is noise in the A matrix
because then if an agent incorrectly interprets a cooperation signal as a cheat
then they will end up in an equilibrium distribution of always cheating 
and always losing p(s = --).

Therefore there is forgiving tit for tat, there is a nonzero probability
that the player will cooperate even if the opponent cheated

This way, in a noisy system, the players will eventually be able to always converge back to p(s = ++)"""

""" p1 = probability that i cooperate if you cheated < 0.5
    p2 = probability that you cooperate if i cheated < 0.5

p(+_ | _-) = p1
p(_+ | -_ ) = p2
p(-_ | _-) = (1-p1)
p(_- | -_) = (1-p2)

p(_- | +_) = 0
p(-_ | _+) = 0

"""

#p(++ | ++ ) = 1
#p( ++ | +- ) = p1
#p( ++ | -+ ) = p2
#p( ++ | -- ) = p1*p2

#p( +- | ++ ) = 0
#p( +- | +- ) = 0
#p( +- | -+ ) = (1-p2)
#p( +- | -- ) = p1*(1-p2)

#p( -+ | ++ ) = 0
#p( -+ | +- ) = (1-p1)
#p( -+ | -+ ) = 0
#p( -+ | -- ) = (1-p1)*p2

#p( -- | ++ ) = 0
#p( -- | +- ) = 0
#p( -- | -+ ) = 0
#p( -- | -- ) = (1-p1)*(1-p2)

p1 = 0.2
p2 = 0.2

B_forgiving_tit_for_tat = np.array([[1,p1,p2,p1*p2],
                                    [0,0,1-p2,p1*(1-p2)],
                                    [0,1-p1,0,(1-p1)*p2],
                                    [0,0,0,(1-p1)*(1-p2)]])


"""In a tit for tat strategy, once an opponent defects, the tit for tat player immediately responds by defecting on the next move.
This has the unfortunate consequence of causing two retaliatory strategies to continuously defect against each other resulting in a poor outcome for both players.
A tit for two tats player will let the first defection go unchallenged as a means to avoid the "death spiral" of the previous example.
If the opponent defects twice in a row, the tit for two tats player will respond by defecting.


This would require 'memory'"""
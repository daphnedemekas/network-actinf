import numpy as np 
from pymdp.maths import softmax
from pymdp import utils
def construct_A(precision_prosocial = 3.0, precision_antisocial = 2.0):
    A1_prosocial = np.zeros((4,4))
    A1_prosocial[:,0] = softmax(precision_prosocial* np.array([1,0,0,0]))
    A1_prosocial[:,1] = softmax(precision_antisocial* np.array([0,1,0,0]))
    A1_prosocial[:,2] = softmax(precision_prosocial* np.array([0,0,1,0]))
    A1_prosocial[:,3] = softmax(precision_antisocial* np.array([0,0,0,1]))


    A1_antisocial = np.zeros((4,4))
    A1_antisocial[:,0] = softmax(precision_antisocial* np.array([1,0,0,0]))
    A1_antisocial[:,1] = softmax(precision_prosocial* np.array([0,1,0,0]))
    A1_antisocial[:,2] = softmax(precision_antisocial* np.array([0,0,1,0]))
    A1_antisocial[:,3] = softmax(precision_prosocial* np.array([0,0,0,1]))
    A = utils.obj_array(1)

    A1 = np.zeros((4,4,2))
    A1[:,:,0] = A1_prosocial
    A1[:,:,1] = A1_antisocial
    A[0] = A1
    return A

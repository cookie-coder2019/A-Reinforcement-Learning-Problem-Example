import numpy as np
from gym import Env, spaces
import random
import math
from scipy.special import logsumexp
from typing import Optional
import scipy.spatial.distance as ssd
import sys

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4


STEP = 1                                  # IN METER
ROTSTEP = 1                               # IN DEGREE -90 -89 -88 ... 0 1 2 ... 88 89 90

AREALENGTH = 4
AREAWIDTH = 4

class MyUAVEnvNActions(Env):
    '''
    Description:
    A UAV moves in a region to provide service to the ground users. 
    The problem is to provide the UAV with best possible Sum-Rate over N steps.

    Best possible Sum-Rate & N steps (less steps)
    
    ### Action Space
    The agent(UAV) takes a 1-element vector for actions.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:

    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP
    - 4: STAY

    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the last position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    For example, the 4x4 map has 16 possible observations.

    ### Reward
    
    ### Starting State
    
    ### Episode Termination
    '''

    def __init__(self):
        self.N_user = 2                                                           # try one user first
        self.Dim_user = 3                                                         # [X, Y, Z = 0]
        self.users = []
        self.users.append(np.array([[1, 1, 0]]))
        self.users.append(np.array([[3, 3, 0]]))
        self.users.append(np.array([[2, 2, 0]]))
        self.users.append(np.array([[1, 2, 0]]))



        self.nrow = int(AREAWIDTH/STEP) #4 #int(AREAWIDTH/STEP)
        self.ncol = int(AREALENGTH/STEP) #4 #int(AREALENGTH/STEP)
        self.nrot = 181 #int(181) #int((90-(-90))/ROTSTEP + 1)         # -90 -45 0 45 90


        self.nA = (5, 181)
        self.nS = self.nrow * self.ncol * self.nrot #int(self.nrow * self.ncol * self.nrot)
        #print('self.nS', self.nS)
        self.action_space = spaces.MultiDiscrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)


        self.initial_state = 0
        
        self.R = {}

        for s in range(int(self.nS)):
            row = {}
            self.R[s] = row
            for ai in range(self.nA[0]):
                for aj in range(self.nA[1]):
                    row[(ai,aj)]= []



        def to_s(row, col, rot):
            return row * self.ncol * self.nrot + col * self.nrot + rot

        def rot2rotstate(rot):  # how about rot is in degree
            rotstate = rot
            return rotstate
        def rotstate2rot(rotstate):          # 0: -90    1: -45    2: 0    3: 45    4: 90
            rot = rotstate
            return rot

        def inc(row, col, rot, a):
            rot = a[1]
            if a[0] == LEFT:
                col = max(col - 1, 0)
            elif a[0] == DOWN:
                row = min(row + 1, self.nrow - 1)
            elif a[0] == RIGHT:
                col = min(col + 1, self.ncol - 1)
            elif a[0] == UP:
                row = max(row - 1, 0)

            return (row, col, rot) # rot in degree

        def update_reward_matrix(row, col, rot, action):
            newrow, newcol, newrot_real = inc(row, col, rot, action)
            newrot_state = rot2rotstate(newrot_real)   # state
            newstate = to_s(newrow, newcol, newrot_state)

            distance = []
            cos_phi = []
            phi = []
            cos_sigma_ori = []
            sigma_ori = []
            sigma_final = []
            cos_sigma = []
            h = []
            dr =[]
            for i in range(4):
                distance.append(ssd.euclidean([newrow*STEP, newcol*STEP, 10],[self.users[i]]))
                cos_phi.append((10)/distance[i])
                phi.append(math.acos(cos_phi[i]))
                cos_sigma_ori.append(cos_phi[i])
                sigma_ori.append(math.acos(cos_sigma_ori[i]))
                sigma_final.append(abs(sigma_ori[i] - newrot_real*(math.pi/180)))
                cos_sigma.append(math.cos(sigma_final[i]))
                h.append((((1+1)*1)/(2*math.pi*(distance[i]*distance[i])))*(((1.5*1.5)/((math.sin(math.pi/3))*(math.sin(math.pi/3)))))*(cos_sigma[i])*cos_phi[i])
                dr.append((20*(10**6))*math.log2(1+ (math.exp(1)/(2*math.pi))*((h[i])**2)))
            reward = sum(dr)            
            return newstate, reward, newrot_real, newrow, newcol #newcol*STEP

        for row in range(int(self.nrow)):
            for col in range(int(self.ncol)):
                for rot in range(int(self.nrot)):   # rotstate 0 1 2 3 4
                    s = to_s(row, col, rot)
                    realrot = rotstate2rot(rot)
                    for ai in range(self.nA[0]):
                        for aj in range(self.nA[1]):
                            Re = self.R[s][(ai,aj)]
                            # rot = rotstate2rot(rot)
                            # print('test??:', *update_reward_matrix(row, col, realrot, (ai,aj)), file=open("Proof_by_exhaustion_new0701.txt", "a"))
                            Re.append((1.0, *update_reward_matrix(row, col, realrot, (ai,aj))))






    def step(self, action):
        #print(action[0])
        #if action.shape == 1:
        #        print('ERROR', action)
        #print('action',action)
        #print('self.s',self.s)
        temp_list = self.R[self.s][(action[0], action[1])]
        #print('temp_list: ', temp_list)
        #temp_list = self.R[self.s][action]
        #print('rot!!!!', temp_list[0][3])
        s = temp_list[0][1]
        r = temp_list[0][2]
        #with open('newtest333.txt','a') as teststates1:
        # with open('newtest555.txt','a') as teststates1:
        #     teststates1.write('rot state:' + str(temp_list[0][3]) + '\n')
        #     teststates1.write('row state:' + str(temp_list[0][4]) + '\n')
        #     teststates1.write('col state:' + str(temp_list[0][5]) + '\n')
        # with open('newtest06290507.txt','a') as teststates1:
        #     teststates1.write('rot in degree:' + str(temp_list[0][3]) + '\n')
        #     teststates1.write('row state:' + str(temp_list[0][4]) + '\n')
        #     teststates1.write('col state:' + str(temp_list[0][5]) + '\n')
        # print('rot state:', temp_list[0][3], file=open("teststates.txt", "a"))
        # print('row state:', temp_list[0][4], file=open("teststates.txt", "a"))
        # print('col state:', temp_list[0][5], file=open("teststates.txt", "a"))
        self.s = s # int(s) works for 2000 times then action just got one, logic not correct, should be s not int(s)
        #print('self.s1',self.s)
        self.lastaction = action
        #file = open('output.txt','a')
        #sys.stdout = file
        #print('action',action)
        #print('self.s',self.s)
        #print('rot!!!!', temp_list[0][3])
        #file.close()
        return (int(s), r , False, {})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = 0
        self.lastaction = None

        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {}
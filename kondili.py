import gym 
import numpy as np
from gym import spaces
import csv
import random
import torch

class KondiliEnv(gym.Env):           #action    #accumulated times
    def __init__(self):   
        self.observation_space = spaces.Dict({"agent" : spaces.Box(0, 1, shape=(2,), dtype=int),
                                              "target": spaces.Box(0, 1, shape=(2,), dtype=int),})
        self.action_space = spaces.Discrete(4)
        self.max_steps = 30
        self.max_val = 1.0
        
    def step1(self, d_action, c_action, node_matrix, times):  
        self.d_action = d_action
        self.c_action = c_action[0]
        obs = node_matrix[0]  
        t = times
        self.r = 0
        self.done = False
        self.available_actions = np.array([1,1,1,1,1,1,1,1,1], dtype=np.int32)
        
        #capacities       
        r1cap  =   80
        r2cap  =   50
        discap =  200
        hcap   =  100
        p1cap  =  300   #max capacity
        p2cap  =  300

        hacap  =  100
        iecap  =  100
        ibccap =  150
        iabcap =  200

        #times
        ht   =  2 #1
        r1t  =  4 #2
        r2t  =  4 #2
        r3t  =  2 #1
        dist =  2 #1
        
        #continuous
        a = np.clip(self.c_action[0], 0.0, 1.0)   #Heater                   100
        b = np.clip(self.c_action[1], 0.0, 1.0)   #reaction1 in reactor 1    80 
        c = np.clip(self.c_action[2], 0.0, 1.0)   #reaction2 in reactor 1    80 
        d = np.clip(self.c_action[3], 0.0, 1.0)   #reaction1 in reactor 2    50
        e = np.clip(self.c_action[4], 0.0, 1.0)   #reaction2 in reactor 2    50
        f = np.clip(self.c_action[5], 0.0, 1.0)   #reaction3 in reactor 1    80  
        g = np.clip(self.c_action[6], 0.0, 1.0)   #reaction3 in reactor 2    50
        h = np.clip(self.c_action[7], 0.0, 1.0)   #Distiller                200

############################################## Move times one step before####################################
        obs[:,-4] = obs[:,-3]
        obs[:,-3] = obs[:,-2]
        obs[:,-2] = obs[:,-1]
############################################## Move products to storage #####################################
        #heater  
        if obs[0][0] ==1:
            t[6] += 1
            obs[0][9] = t[6]/ht
        if t[6] > ht:
            obs[5][5] += obs[0][5] * hcap / hacap
            obs[0][:] = 0
            t[6] = 0
            
        #reaction 1 in reactor 1    
        if obs[1][1] == 1:
            t[0] += 1
            obs[1][9] = t[0]/r1t
            if t[0] > r1t:
                obs[6][5] += obs[1][5] * r1cap / ibccap
                obs[1][:] = 0 
                t[0] = 0
            
        #reaction 2 in reactor 1    
        if obs[2][1] == 1:
            t[1] += 1
            obs[2][9] = t[1]/r2t
            if t[1] > r2t:
                obs[7][5] += obs[2][5] * r1cap *0.6 / iabcap
                obs[10][5] += obs[2][5] * r1cap *0.4 / p1cap
                obs[2][:] = 0
                t[1] = 0
        
        #reaction 3 in reactor 1    
        if obs[3][1] == 1:
            t[2] += 1
            obs[3][9] = t[2]/r3t
            if t[2] > r3t:
                obs[8][5] += obs[3][5] * r1cap / iecap
                obs[3][:] = 0
                t[2] = 0
        
        #reaction 1 in reactor 2    ok  
        if obs[11][2] == 1:
            t[3] += 1
            obs[11][9] = t[3]/r1t
            if t[3] > r1t:
                obs[6][5] += obs[11][5] * r2cap / ibccap
                obs[11][:] = 0
                t[3] = 0
            
        #reaction 2 in reactor 2     ok 
        if obs[12][2] == 1:
            t[4] += 1
            obs[12][9] = t[4]/r2t
            if t[4] > r2t:
                obs[7][5]  += obs[2][5] * r2cap * 0.6 / iabcap
                obs[10][5] += obs[2][5] * r2cap * 0.4 / p1cap
                obs[12][:] = 0
                t[4] = 0
            
        #reaction 3 in reactor 2  ok
        if obs[13][2] == 1:
            t[5] += 1
            obs[13][9] = t[5]/r3t
            if t[5] > r3t:
                obs[8][5] += obs[3][5] * r2cap / iecap
                obs[13][:] = 0
                t[5] = 0
        
        #distiller      
        if obs[4][3] == 1:
            t[7] += 1
            obs[4][9] = t[7]/dist
            if t[7] > dist:
                obs[9][5] += obs[4][5] * discap * 0.9 / p2cap
                obs[7][5] += obs[4][5] * discap * 0.1 / iabcap
                obs[4][:] = 0
                t[7] = 0        
        
############################################## overloading storage ###################################        
        if obs[5][5] > 1:             
            obs[5][5] = 1
            
        if obs[6][5] > 1:
            obs[6][5] = 1 
            
        if obs[7][5] > 1:
            obs[7][5] = 1
        
        if obs[8][5] > 1:
            obs[8][5] = 1

############################################## New actions ##########################################         
        #choosing the heater        
        if self.d_action == [0] and a > 0:
            wa = 50 * a
            if obs[0][0] == 0:                              #if the machine is empty
                self.r += wa
                if obs[5][5]*hacap + a*hcap <= hacap:       #if there is space after finishing
                    obs[0][5] = a
                    obs[0][0] = 1
                    obs[0][4] = hcap/p1cap
                    t[6] = 1
                    obs[0][9] = t[6]/ht
                    self.r += wa 
                else:
                    self.r -= wa + ( obs[5][5]*hacap + a*hcap - hacap) #/ hcap)
            else:
                self.r = -100

        #choosing reaction1 in reactor 1    
        if self.d_action == [1] and b > 0:
            wb = 100 * b 
            if obs[1][1] + obs[2][1] + obs[3][1] == 0:
                self.r += wb
                if obs[6][5]*r1cap + b*r1cap <= ibccap:     #if there is enough space to store it
                    obs[1][5] = b
                    obs[1][1] = 1
                    obs[1][4] = r1cap/p1cap
                    t[0] = 1
                    obs[1][9] = t[0]/r1t
                    self.r += wb
                else:
                    self.r -= wb + (obs[6][5]*ibccap + b*r1cap - ibccap) #/ r1cap)
            else:
                self.r = -100

        #chossing reaction2 in reactor 1    
        if self.d_action == [2] and c > 0:
            wc = 125 * c 
            if obs[1][1] + obs[2][1] + obs[3][1] == 0:                           # if the reactor is empty
                self.r += wc
                if obs[5][5]*hacap >= c*r1cap*0.4 and obs[6][5]*ibccap >= c*r1cap*0.6:  # if there is enough material 
                    self.r += wc
                    if obs[7][5]*iabcap + c*r1cap*0.6 <= iabcap :                # if there is enough place to store it
                        obs[5][5] -= c*r1cap*0.4 / hacap
                        obs[6][5] -= c*r1cap*0.6 / ibccap
                        obs[2][5] = c
                        obs[2][1] = 1
                        obs[2][4] = r1cap/p1cap
                        t[1] = 1
                        obs[2][9] = t[1]/r2t
                        self.r += wc
                    else:
                        self.r -= 2*wc +  (obs[7][5]*iabcap + c*r1cap*0.6 - iabcap) #( r1cap*0.6))
                else:
                    self.r -= wc + np.min(( (c*r1cap*0.4 - obs[5][5]*hacap) , (c*r1cap*0.6 - obs[6][5]*ibccap) )) #( (c*r1cap*0.4 - obs[4])/(r1cap*0.4) , (c*r1cap*0.6 - obs[5])/(r1cap*0.6) )
            else:
                self.r = -100                   

        #choosing reaction 3 in reactor 1          
        if self.d_action == [5] and f > 0:          
            wf = 300 * f
            if obs[1][1] + obs[2][1] + obs[3][1] == 0:      # if the reacotr is not busy
                self.r += wf
                if obs[7][5]*iabcap +1e-2 >= f*r1cap*0.8 :   # if there is enough material
                    self.r += wf
                    if obs[8][5]*iecap + f*r1cap <= iecap : # if there is enough place to store
                        obs[7][5] -= f*r1cap*0.8 / iabcap
                        obs[3][5] = f
                        obs[3][2] = 1
                        obs[3][4] = r1cap/p1cap
                        t[2] = 1
                        obs[3][9] = t[2] / r3t
                        self.r += wf
                    else:
                        self.r -= 2*wf + (obs[8][5]*iecap + f*r1cap - iecap)#/r1cap)
                else:
                    self.r -= wf + (f*r1cap*0.8 - obs[7][5]*iabcap)         #/(r1cap*0.8)
            else:
                self.r = -100

        #choosing reaction1 in reactor 2        ok
        if self.d_action == [3] and d > 0:
            wd = 100* d 
            if obs[11][2] + obs[12][2] + obs[13][2] == 0:      # if the reactor is not busy
                self.r += wd
                if obs[6][5]*ibccap + d*r2cap <= ibccap:    # if there is enough storage
                    obs[11][5] = d
                    obs[11][2] = 1
                    obs[11][4] = r2cap/p1cap 
                    t[3] = 1
                    obs[11][9] = t[3]/r1t
                    self.r += wd 
                else:
                    self.r -= wd + (obs[6][5]*ibccap + d*r2cap - ibccap)#/r2cap)
            else:
                self.r = -100

        #chossing reaction2 in reactor 2     ok  
        if self.d_action == [4] and e > 0:
            we = 125 * e
            if obs[11][2] + obs[12][2] + obs[13][2] == 0:
                self.r += we
                if obs[5][5]*hacap >= e*r2cap*0.4 and obs[6][5]*ibccap >= e*r2cap*0.6:  
                    self.r += we
                    if obs[7][5]*iabcap + e*r2cap*0.6 <= iabcap:
                        obs[5][5] -= e*r2cap*0.4 / hacap
                        obs[6][5] -= e*r2cap*0.6 / ibccap
                        obs[12][5] = e
                        obs[12][2] = 1
                        obs[12][4] = r2cap/p1cap
                        t[4] = 1
                        obs[12][9] = t[4]/r2t
                        self.r += we
                    else:
                        self.r -= 2*we + (e*r2cap*0.6 + obs[7][5]*iabcap - iabcap)#/(r2cap*0.6))
                else:
                    self.r -= we + np.min(( (e*r2cap*0.4 - obs[5][5]*hacap) , (e*r2cap*0.6 - obs[6][5]*ibccap)  )) #( (e*r2cap*0.4 - obs[4])/(r2cap*0.4) , (e*r2cap*0.6 - obs[5])/(r2cap*0.6)  )
            else:
                self.r = -100   

        #choosing reaction 3 in reactor 2    ok
        if self.d_action == [6] and g > 0:
            wg = 300 * g 
            if obs[11][2] + obs[12][2] + obs[13][2] == 0:      # if the reactor is not busy
                self.r += wg
                if obs[7][5]*iabcap >= g*r2cap*0.8:         # if there is enough material
                    self.r += wg
                    if obs[8][5]*iecap + g*r2cap <= iecap:  # if there is enough space for store
                        obs[7][5] -= g*r2cap*0.8 / iabcap    
                        obs[13][5] = g
                        obs[13][2] =1 
                        obs[13][4] = r2cap/p1cap
                        t[5] = 1
                        obs[13][9] = t[5]/r3t
                        self.r += wg
                    else:
                        self.r -= 2*wg + (g*r2cap + obs[8][5]*iecap - iecap)#/r2cap)
                else:
                    self.r -= wg + (g*r2cap*0.8 - obs[7][5]*iabcap)#/(r2cap*0.8)
            else:
                self.r = -100

        #choosing the distiller     
        if self.d_action == [7] and h > 0:
            wh = 1600 * h
            if obs[4][3] == 0:                          # if dest is not busy
                self.r += wh
                if obs[8][5]*iecap >= h*discap:         # if there is enough material
                    self.r += wh
                    if obs[7][5]*iabcap + h*discap*0.1 <= iabcap :      # if there is enough place to store
                        obs[8][5] -= h*discap / iecap
                        obs[4][5] = h
                        obs[4][3] = 1
                        obs[4][4] = discap/p1cap
                        t[7] = 1
                        obs[4][9] = t[7]/dist
                        self.r += wh
                    else:
                        self.r -= 2*wh + (h*discap*0.1 + obs[7][5]*iabcap - iabcap)#/(discap*0.1))
                else:
                    self.r -= wh + (h*discap - obs[8][5]*iecap)#/discap
            else:
                self.r = -100

        ################################# Add capacity ################################
        if obs[5][5] > 0 :
            obs[5][4] = hacap/p1cap
        else: obs[5][4] = 0

        if obs[6][5] > 0:
            obs[6][4] = ibccap/p1cap
        else: obs[6][4] = 0

        if obs[7][5] > 0:
            obs[7][4] = iabcap/p1cap
        else: obs[7][4] = 0

        if obs[8][5] > 0:
            obs[8][4] = iecap/p1cap
        else: obs[8][4] = 0

        if obs[9][5] > 0:
            obs[9][4] = 1
        else: obs[9][4] = 0

        if obs[10][5] > 0:
            obs[10][4] = 1
        else: obs[10][4] = 0
        ################################# Masking ####################################
        #if t[8] == 0:
        #    self.available_actions = np.array([1,0,0,0,0,0,0,0,0], dtype=np.int32)    #[0,1,0,0,0,0,0,0,0]
        #if t[8] == 1:
        #    self.available_actions = np.array([0,0,1,0,0,0,0,0,0], dtype=np.int32)
        #Prevent action 0
        #if obs[4] > 32:
        #    self.available_actions[0] = 0
        #Prevent Action 0

        '''
        tot_h = obs[0] + obs[4] +  obs[8]
        if t[1] != 0:
            tot_h+= 0.4*obs[1]
        if t[4] != 0:
            tot_h += 0.4*obs[2]
        if  tot_h> 170:
            self.available_actions[0] = 0
        # Prevent Action 2
        if obs[4] < 32 or obs[5] < 48:       #si no hay suficiente HotA ni IntBC para reactor 1 al 100% (80)
            self.available_actions[2] = 0
        # Prevent Action 4
        if obs[4] < 20 or obs[5] < 30:       #si no hay suficiente HotA ni IntBC para reactor 2 al 100% (50)
            self.available_actions[4] = 0 
        # Prevent Action 5
        if obs[6] < 64:                      #si no hay suficiente IntAB para reactor 1 al 100%         (80)
            self.available_actions[5] = 0
        # Prevent Action 6
        if obs[6] < 40:                      #si no hay suficiente IntAB para reactor 2 al 100%         (50)
            self.available_actions[6] = 0
        # Prevent Action 7
        if obs[7] < 10:                      #si no hay suficiente impE para separador
            self.available_actions[7] = 0  
        
        # Prevent Actions from occupied machines    
        if obs[11] > 1:  #si el reactor 1 estara ocupado en el proximo state
            self.available_actions[1] = 0    #si es 0 o 1 significa que esta desocupado o lo estara
            self.available_actions[2] = 0
            self.available_actions[5] = 0
        if obs[11] > 1:                      #si el reactor 2 estara ocupado en el proximo state
            self.available_actions[3] = 0    #si es 0 o 1 significa que esta desocupado o lo estara
            self.available_actions[4] = 0
            self.available_actions[6] = 0
        '''

        if obs[1][1] == 1:
            if obs[1][9] < 0.75:
                self.available_actions[[1,2,5]] = 0
        if obs[2][1] == 1:
            if obs[2][9] < 0.75:
                self.available_actions[[1,2,5]] = 0
        if obs[3][1] == 1:
            if obs[3][9] < 0.75:
                self.available_actions[[1,2,5]] = 0

        if obs[11][2] == 1:
            if obs[1][9] < 0.75:
                self.available_actions[[3,4,6]] = 0
        if obs[12][2] == 1:
            if obs[2][9] < 0.75:
                self.available_actions[[3,4,6]] = 0
        if obs[13][2] == 1:
            if obs[3][9] < 0.75:
                self.available_actions[[3,4,6]] = 0

        
        if t[8] == self.max_steps:  #for the final step we see the product 
            self.done = True     
        t[8] += 1
        #obs[-1] = t[8]    
        
        # Time restrictions
        if t[8] >= 27:
            self.available_actions[[5,6]] = 0
        if t[8] >= 23:
            self.available_actions[[2,4]] = 0
        if t[8] >= 19:
            self.available_actions[[1,3]] = 0
        if t[8] >= 17:
            self.available_actions[0] = 0
        if t[8] >= 29:
            self.available_actions[7] = 0 
         
        ##############################################################################    
        #if t[8] == self.max_steps:  #for the final step we see the product 
        #    self.done = True           
        #t_factor = (self.max_steps-t[8])/self.max_steps

        reward = self.r #+ t_factor*obs[4]/100 + t_factor*1.7*obs[5]/100 + 2*((1/0.4)*obs[8])/100 + 2.5*obs[7]/100 \
            #+ 5*(1/0.9)*(obs[9])/100 + 5*(obs[3])/100 #+ 25*obs[3] + 2*obs[7] #2
        #normalize reward
        #min_reward = -200 #-200
        #max_reward = 400  #500
        #n_reward = (reward - min_reward) / (max_reward - min_reward) * 2 - 1

        #t[8] += 1
        #obs[-1] = t[8]
        return obs, reward, self.done, t, self.available_actions
        ##############################################################################        
    def reset(self):
        initial_states = torch.zeros((14,10), dtype=torch.float32)
        self.done = False     
        return initial_states
    
    def reset_m(self,obs): 
        self.available_inactions = np.array([1,0,0,0,0,0,0,0,0], dtype=np.int32) 
        return self.available_inactions
    
    def action_mask(self):
        mask_vec = np.array(self.available_actions, dtype=bool)
        return mask_vec
    
    def render (self):
        pass

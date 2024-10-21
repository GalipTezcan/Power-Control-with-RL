import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, random_state):
        super().__init__()
        self.seed=random_state
        
        random.seed(random_state)
        np.random.seed(random_state)
        self.state_size=3
        self.retainability=[]
        self.epsoide_retainability=[]
        self.SINR_MIN = -3 #dB  
        self.baseline_SINR_dB = 4.0
        self.final_SINR_dB = self.baseline_SINR_dB + 2.0 # this is the improvement
        #MAX_EPISODES = 707 # successful ones [85, 115, 129, 258, 259, 284, 285, 286, 707]
        self.N_interferers = 4 # this is a hardcoded parameter: 4 base stations.

        self.L_geom = 10. #  meters

        self.pt_max = 2 # in Watts

        # all in dB here
        self.g_ant = 2
        self.ins_loss = 0.5
        self.g_ue = 0
        self.f = 2600 
        self.h_R = 1.5 # human UE
        self.h_B = 3 # base station small cell height
        self.NRB = 100 # Number of PRBs in the 20 MHz channel-- needs to be revalidated.
        self.B = 20e6 # 20 MHz
        self.T = 290 # Kelvins/
        self.K = 1.38e-23
        self.num_antenna_streams = None # Not needed
        self.N0 = self.K*self.T*self.B*1e3 # Thermal noise.  Assume Noise Figure = 0 dB.

        # This formula is from Computation of contribution of action νin= 2 in paper.

        numerator =  10**((self.pt_max - np.log10(self.NRB) + self.g_ant - self.ins_loss)/10)
        denom = (self.N0 + (self.N_interferers - 1) * self.pt_max)
        SINR_due_to_neighbor_loss= 10*np.log10(numerator / denom)

        ################ Actions for Power Control ###################
        # 0- Cluster is normal
        # 1- Feeder fault alarm (3 dB)
        # 2- Neighboring cell down
        # 3- VSWR out of range alarm: https://antennatestlab.com/antenna-education-tutorials/return-loss-vswr-explained
        # 4- Clear 1 ( - dB losses)
        # 5- Clear 2
        # 6- Clear 3
            
        
        state_count = 3 # ok we agree on this
        action_count_a = 11 # this is the upper index of player A index.
        action_count_b = 5 # this is per player_B_contribs.

        # Network
        player_A_scenario_0_SINR_dB = self.average_SINR_dB(g_ant=16, num_users=10, faulty_feeder_loss=0.0, beamforming=False, random_state=self.seed)
        player_A_scenario_1_SINR_dB = self.average_SINR_dB(g_ant=16, num_users=10, faulty_feeder_loss=3.0, beamforming=False, random_state=self.seed)
        player_A_scenario_2_SINR_dB = SINR_due_to_neighbor_loss
        player_A_scenario_3_SINR_dB = self.average_SINR_dB(g_ant=16, num_users=10, faulty_feeder_loss=5, beamforming=False, random_state=self.seed) #  VSWR of 1.4:1 too bad.  We want VSWR 1.2:1.  This is 5 dB

        player_A_SINRs = [0, player_A_scenario_1_SINR_dB - player_A_scenario_0_SINR_dB, player_A_scenario_2_SINR_dB, player_A_scenario_3_SINR_dB - player_A_scenario_0_SINR_dB]
        player_A_contribs = np.append(0-np.array(player_A_SINRs), np.array(player_A_SINRs))
        self.player_B_contribs = np.array([0, -3, -1, 1, 3]) # TPCs (up to 2 TPCs per TTI)
        self.player_A_contribs = np.append([0,0,0], player_A_contribs)
        
        self.alarm_reg = [0,0,0]
        self.timestep_index=0
        self.max_timestep=20
        self.action_space = spaces.Discrete(5)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Discrete(3)

        self.reset()

    def step(self, action):
        
        network_issue = self.get_A_contrib()
        self.cell_score += network_issue #player_A_contribs[np.random.randint(action_count_a)]
        # Check if action is integer
        if isinstance(action,tuple):
                action,_=action
        if isinstance(action, np.ndarray) and action.shape!=():
            action = action[0]
        
         # actions: 0 nothing
         # actions: 1, 2 this is a power down
         #          : 3, 4 this is a power up

        # The next states based on the action
        if action == 0:
            observation = 0  # do nothing
            reward = -100 
        elif action == 1 or action == 2:
            observation = 2  # SINR has reduced (either by -1 or -3), state = 2
            reward = -1
            #print(action, self.observation, reward)
        elif action == 3 or action == 4:
            observation = 1  # SINR has increased (either by 1 or 3), state = 1
            reward = 1
           # print(action, self.observation, reward)
        else:
            observation = 0  # do nothing
            reward = -100

        power_command = self.player_B_contribs[action]

        pt_new = self.pt_current*(10 ** (power_command / 10.)) # the current ptransmit in mW due to PC
                    # ptransmit cannot exceed pt, max.
        if (pt_new <= self.pt_max):
            self.pt_current = pt_new
            self.cell_score += power_command   
        
        self.retainability.append(np.round(self.cell_score, 2))         
        truncated = (self.cell_score < self.SINR_MIN)
        succes = (self.cell_score >= self.final_SINR_dB)
        terminated=False
        if succes:
                if self.timestep_index < 15: # premature ending -- cannot finish sooner than 15 episodes
                    truncated = True
                    terminated = False
                else:                       # ending within time.
                   terminated = True
                   reward = 2
                   truncated = False
        if truncated == True:
            reward = -100
        #if terminated or truncated:
        #    for _ in range(self.max_timestep-1-self.timestep_index):
        #        self.retainability.append(np.round(self.cell_score, 2))
        self.timestep_index+=1
        self.epsoide_retainability.append(self.cell_score)
        reward2=np.round(1.-(sum(1 for i in self.epsoide_retainability if i <= 0) / len(self.epsoide_retainability)),2)
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self.cell_score = self.baseline_SINR_dB
        self.pt_current=0.1
        self.alarm_reg = [0,0,0]
        self.timestep_index=0
        self.epsoide_retainability=[]
        return 0,{}

    def render(self):
        ...

    def close(self):
        ...
    def retainability_score(self):
        #print(self.retainability[0:20])
        #print(len(self.retainability))
        dcr =sum(1 for i in self.retainability if i <= 0) / len(self.retainability)
        return 1. - dcr
    def cost231(self,distance, f, h_R, h_B):
        C = 0
        a = (1.1 * np.log10(f) - 0.7)*h_R - (1.56*np.log10(f) - 0.8)
        L = []
        for d in distance:
            L.append(46.3 + 33.9 * np.log10(f) + 13.82 * np.log10(h_B) - a + (44.9 - 6.55 * np.log10(h_B)) * np.log10(d) + C)
        
        return L # in dB



    def compute_interference_power(self,base_station_id, UE_x, UE_y):
        # Returns the received interference power in mW as measured by the UE at location (x,y)
        # The four interfering base stations.
        if (base_station_id == 0):
            X_bs = -self.L_geom
            Y_bs = 0
        if (base_station_id == 1):
            X_bs = 0
            Y_bs = self.L_geom
        if (base_station_id == 2):
            X_bs = 0
            Y_bs = -self.L_geom
        if (base_station_id == 3):
            X_bs = self.L_geom
            Y_bs = 0
            
        # Distances in kilometers.
        UE_x = np.array(UE_x)
        UE_y = np.array(UE_y)
    
        dist = np.power((np.power(X_bs-UE_x, 2) + np.power(Y_bs-UE_y, 2)), 0.5) / 1000.
        recv_power = self.pt_max * 1e3 / np.array([10 ** (l / 10.) for l in self.cost231(dist, self.f, self.h_R, self.h_B)]) # in mW
        
        average_recv_power = sum(recv_power) / len(recv_power)
        return average_recv_power

    def average_SINR_dB(self,random_state, g_ant=2, num_users=50, load=0.7, faulty_feeder_loss=0.0, beamforming=False, plot=False):
        np.random.seed(random_state)
        n = np.random.poisson(lam=num_users, size=None) # the Poisson random variable (i.e., the number of points inside ~ Poi(lambda))

        dX = self.L_geom
        dY = self.L_geom
        
        u_1 = np.random.uniform(0.0, dX, n) # generate n uniformly distributed points 
        u_2 = np.random.uniform(0.0, dY, n) # generate another n uniformly distributed points 
        
        # Now put a transmitter at point center
        X_bs = dX / 2.
        Y_bs = dY / 2.
            
        ######
        # Cartesian sampling
        #####
            
        # Distances in kilometers.
        dist = np.power((np.power(X_bs-u_1, 2) + np.power(Y_bs-u_2, 2)), 0.5) / 1000. #LA.norm((X_bs - u_1, Y_bs - u_2), axis=0) / 1000.

        path_loss = np.array(self.cost231(dist, self.f, self.h_R, self.h_B))
        
        ptmax = 10*np.log10(self.pt_max * 1e3) # to dBm
        pt = ptmax - np.log10(self.NRB) + g_ant - self.ins_loss - faulty_feeder_loss #+ 10 * np.log10(num_antenna_streams) # - np.log10(12)
        pr_RB = pt - path_loss + self.g_ue # received reference element power (almost RSRP depending on # of antenna ports) in dBm.
    
        pr_RB_mW = 10 ** (pr_RB / 10) # in mW the total power of the PRB used to transmit packets
        
        # Compute received SINR
        # Assumptions:
        # Signaling overhead is zero.  That is, all PRBs are data.
        
        SINR = []

        # Generate some thermal noise
        SINR = []
        for i in np.arange(len(dist)): # dist is a distance vector of UE i
            ICI = 0 # interference on the ith user
            for j in np.arange(self.N_interferers):
                ICI += self.compute_interference_power(j, u_1, u_2)

            SINR_i = pr_RB_mW[i] / (self.N0 + ICI)
            SINR.append(SINR_i)
        
        SINR_average_dB = 10 * np.log10(np.mean(SINR))
        return SINR_average_dB
    def get_A_contrib(self):
        # Draw a number at random
        n = np.random.randint(11)

        # if n is less than 4, then it is a Normal action, let it go through..  
        if n < 4:
            return self.player_A_contribs[n]
        
        if n < 7: # this is a clear alarm  (8,9,10)--- only clear it if the alarm has been set, otherwise, return no change
            if (self.alarm_reg[n - 4] == 1):
                self.alarm_reg[n - 4] = 0 # alarm has been cleared.
                return self.player_A_contribs[n]       
            else:
                return 0
        elif n>7: 
            # an alarm
            if (self.alarm_reg[n - 8] == 0):
                self.alarm_reg[n - 8] = 1 #  set up alarm in register.            
                return self.player_A_contribs[n]
            else:
                return 0

        return self.player_A_contribs[n]

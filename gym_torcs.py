import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time
import sys


class TorcsEnv:
    terminal_judge_start = 128 #1000  # If after 100 timestep still no progress, terminated
    speed_ratio = 1
    termination_limit_progress = 5/speed_ratio  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50 
    
    initial_reset = True
    
    # Constants for reward shaping
    PROGRESS_REWARD = 10.0
    COLLISION_PENALTY = -100.0
    SPEED_REWARD_MULTIPLIER = 2.0
    TRACK_CENTER_REWARD = 1.0
    LAP_COMPLETION_REWARD = 100.0
    TIME_PENALTY = -0.20

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True
        self.oot_count = 0
        self.no_prog_count = 0
        self.old_distRaced = 0.0
        self.old_distFromStart = 0.0
        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
        client = self.client
        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d
        self.end_type = 0
        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            #sys.exit()

            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])/self.speed_ratio
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])
        episode_terminate = False
        
        # Calculate the change in distance covered from the previous step
        distance_covered = obs['distFromStart'] - self.old_distFromStart
        #distance_covered = obs['distRaced'] - self.old_distRaced

        #progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        #reward = progress
        
        # Encourage higher speeds, but penalize excessive speed
        reward = sp * self.SPEED_REWARD_MULTIPLIER

        # Encourage staying close to the center of the track
        reward += sp * self.TRACK_CENTER_REWARD * (1 - abs(obs['trackPos']))
        
        # Encourage progress on the track
        reward += distance_covered * self.PROGRESS_REWARD
        
        
        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            print("Car was damaged !!!")
            reward += self.COLLISION_PENALTY
            #episode_terminate = True
            #client.R.d['meta'] = True
            #self.end_type = 1
            
        # Termination judgement #########################
        
        if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
            print("***"*10)
            print("***"*10)
            print("Out of track ")
            print("***"*10)
            print("***"*10)
            #reward += -200*np.abs(np.sin(obs['angle']/2)) #out of track penalty
            #reward += -150*(1-np.exp(-np.abs(8*(obs['angle'])/np.pi)))
            episode_terminate = True
            self.oot_count +=1
            if self.oot_count >6:
                reward = -200
                client.R.d['meta'] = True
                self.end_type = 2

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if sp < self.termination_limit_progress:
                print("***"*10)
                print("***"*10)
                print("No progress", sp)
                print("***"*10)
                print("***"*10)
                self.no_prog_count += 1
                episode_terminate = True
                if self.no_prog_count >9:
                    reward += -(100)
                    client.R.d['meta'] = True
                    self.end_type = 3
                    
        if episode_terminate == False:
            self.oot_count += -2
            self.no_prog_count += -2
            if self.oot_count < 0:
                self.oot_count = 0
            if self.no_prog_count < 0:
                self.no_prog_count = 0


        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            reward += -200
            print("Wrong direction")
            episode_terminate = True
            client.R.d['meta'] = True
            self.end_type = 4
        
        if obs['lastLapTime'] > 0:
            print("...LAP FINISHED...")
            reward += self.LAP_COMPLETION_REWARD
            episode_terminate = True
            client.R.d['meta'] = True
            self.end_type = 5

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()
            
        # Penalize for taking too much time
        reward += self.TIME_PENALTY
        self.old_distRaced = obs['distRaced']
        self.old_distFromStart = obs['distFromStart']
        self.time_step += 1
        #normalized_reward = (reward - 76.9) / 46.4
        return self.get_obs(), reward, client.R.d['meta'], {}, self.end_type

    def reset(self, relaunch=False):
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec  
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track', 
                     'trackPos',
                     'wheelSpinVel',
                     'distRaced',
                     'lastLapTime',
                     'curLapTime',
                     'distFromStart']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               distRaced=np.array(raw_obs['distRaced'], dtype=np.float32),
                               lastLapTime=np.array(raw_obs['lastLapTime'], dtype=np.float32),
                               curLapTime=np.array(raw_obs['curLapTime'], dtype=np.float32),
                               distFromStart=np.array(raw_obs['distFromStart'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img',
                     'distRaced',
                     'lastLapTime',
                     'curLapTime',
                     'distFromStart']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]]) 

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb,
                               distRaced=np.array(raw_obs['distRaced'], dtype=np.float32),
                               lastLapTime=np.array(raw_obs['lastLapTime'], dtype=np.float32),
                               curLapTime=np.array(raw_obs['curLapTime'], dtype=np.float32),
                               distFromStart=np.array(raw_obs['distFromStart'], dtype=np.float32))

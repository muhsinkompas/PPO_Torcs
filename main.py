import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from class_ppo import *
from gym_torcs import TorcsEnv
import rwfile as rw
import wOutputToCsv as w_Out
import os

#----------------------------------------------------------------------------------------

EP_MAX = 4000
EP_LEN = 4000
GAMMA = 0.95


A_LR = 1e-4
C_LR = 1e-4

BATCH = 64
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 29, 3
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=1.0),   # KL penalty; lam is actually beta from the PPO paper
    dict(name='clip', epsilon=0.20),           # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


# train_test = 0 for train; =1 for test
train_test = 1
# irestart = 0 for fresh restart; =1 for restart from ckpt file
irestart = 1
iter_num = 0
relaunch_le = 25 #relaunch for leak memory error.

if (irestart == 0):
    iter_num = 0

#----------------------------------------------------------------------------------------

sess = tf.Session()

ppo = PPO(sess, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, METHOD)

saver = tf.train.Saver(max_to_keep=50)

env = TorcsEnv(vision=False, throttle=True, gear_change=False)

### ----------------------------------------------------------------------- ###

file_path = 'Best/bestlaptime.csv'
r_w = rw.RW(file_path)
best_lap_time = r_w.read_numpy_array_from_csv()
print(best_lap_time)
w_csv = w_Out.OW(csv_path = 'OutputCsv/output.csv',headers = ['ep', 'step', 'a_1', 'a_2', 'a_3' , 'reward', 
                                                              's_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10',
                                                              's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 's_20',
                                                              's_21', 's_22', 's_23', 's_24', 's_25', 's_26', 's_27', 's_28', 's_29', 
                                                              'end_type', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'distRaced', 'distFromStart',
                                                              'curLapTime', 'lastLapTime'])
w_total_csv = w_Out.OW(csv_path = 'OutputCsv/output_total.csv',headers = ['ep', 'step', 'end_type', 'col_count', 'oot_count', 'np_count', 
                                                                          'wrong_direction', 'speedX', 'distRaced', 'distFromStart', 'last_lap_distance', 
                                                                          'curLapTime', 'lastLapTime', 'total_reward'])
w_event_csv = w_Out.OW(csv_path = 'OutputCsv/event_history.csv',headers = ['ep', 'step', 'col_count', 'oot_count', 'np_count', 
                                                                          'wrong_direction', 'distFromStart'])

### ----------------------------------------------------------------------- ###
#actor_losses = []
#critic_losses = []
steps = []
#actor_loss, crit_loss = [0.0 , 0.0]

### ----------------------------------------------------------------------- ###

if (train_test == 0 and irestart == 0):
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, "weights/model.ckpt")  


for ep in range(iter_num, EP_MAX):

    print("-"*50)
    print("episode: ", ep)

    if np.mod(ep, relaunch_le) == 0:
        ob = env.reset(relaunch=True)   #relaunch TORCS every N episode because of the memory leak error
    else:
        ob = env.reset()

    s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    buffer_s, buffer_a, buffer_r = [], [], []
    last_lap_distance = 0
    total_reward = 0
    event_counts = np.array([0, 0, 0, 0])
    event_list = np.array([0, 0, 0, 0, 0, 0, 0])
    
    for t in range(EP_LEN):    # in one episode
       
        a = ppo.choose_action(s)
        
        a[0] = np.clip(a[0],-1.0,1.0)
        a[1] = np.clip(a[1],0.0,1.0)
        a[2] = np.clip(a[2],0.0,1.0)  

        ob, r, done, _, end_type, event_buff = env.step(a)
        event_counts = event_counts + event_buff
        if np.sum(event_buff) > 0:
            event_list_buff = np.hstack((i, j, event_buff, ob.distFromStart))
            event_list = np.vstack((event_list, event_list_buff))
        
        s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))  


        if (train_test == 0):
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)    

        s = s_
        total_reward += r

        if (train_test == 0):
            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN-1 or done == True:
            #if t == EP_LEN-1 or done == True:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs = np.array(np.vstack(buffer_s))
                ba = np.array(np.vstack(buffer_a))  
                br = np.array(discounted_r)[:, np.newaxis]

                buffer_s, buffer_a, buffer_r = [], [], []

                print("ppo update")
                ppo.update(bs, ba, br)
                #actor_loss, crit_loss = ppo.update(bs, ba, br)
                
        ### LAST LAP TIME ### CHECK HERE
        if ob.lastLapTime > 0:
            print("lap time is : ",ob.lastLapTime)
            if (ob.lastLapTime < best_lap_time) and (train_test==0):
                best_lap_time = ob.lastLapTime
                r_w.write_numpy_array_to_csv(best_lap_time)
                saver.save(sess, "Best/model.ckpt")
                print("Best Lap Time is updated.")
                print("saving Best model")
                
        print("="*100)
        print("--- Episode : {:<4}\tActions ".format(ep)+ np.array2string(a, formatter={'float_kind': '{0:.3f}'.format})+"\tReward : {:8.4f}".format(total_reward)+" ---")
        print("="*100)
        
        if ob.distFromStart > last_lap_distance:
            last_lap_distance = ob.distFromStart
        
        #----------------------------------------------------------------------------------------------------------------
        # Saving outputs to csv file
        # print("saving csv")
        #### ADD LOSS HERE
        output_csv = np.hstack((ep, t, a, r, s, end_type, ob.focus, ob.distRaced, ob.distFromStart, ob.curLapTime, ob.lastLapTime))
        w_csv.append_numpy_array_to_csv(np.matrix(output_csv))
        #----------------------------------------------------------------------------------------------------------------

        if (done  == True):
            break
    
    #actor_losses.append(actor_loss)
    #critic_losses.append(crit_loss)
    steps.append(ep)
    ### Saving total outputs for each episode --------------------------------------- ###
    output_total_csv = np.hstack((ep, t, end_type, event_counts, ob.speedX, ob.distRaced, ob.distFromStart, last_lap_distance, ob.curLapTime, ob.lastLapTime, total_reward))
    w_total_csv.append_numpy_array_to_csv(np.matrix(output_total_csv))
    w_event_csv.append_numpy_array_to_csv(np.matrix(event_list))
    ### ----------------------------------------------------------------------------- ###
    
    

    if (train_test == 0):
        with open("performance.txt", "a") as myfile:
            myfile.write(str(ep) + " " + str(t) + " " + str(round(total_reward,4)) + "\n")

    if np.mod(ep, 100) == 99:
        if (train_test ==0):
            file_name = 'Models/'+str(ep+1)
            if os.path.isdir(file_name) ==False:
                os.mkdir(file_name)
            model_name = file_name+'/model.ckpt'
            print("saving model")
            saver.save(sess, model_name)


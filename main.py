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
EP_LEN = 2000
GAMMA = 0.95


A_LR = 3e-4
C_LR = 1e-3

BATCH = 64
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 29, 3
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=1.0),   # KL penalty; lam is actually beta from the PPO paper
    dict(name='clip', epsilon=0.15),           # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


# train_test = 0 for train; =1 for test
train_test = 0
# irestart = 0 for fresh restart; =1 for restart from ckpt file
irestart = 0
iter_num = 0
relaunch_le = 50 #relaunch for leak memory error.

if (irestart == 0):
    iter_num = 0

#----------------------------------------------------------------------------------------

sess = tf.Session()

ppo = PPO(sess, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, METHOD)

saver = tf.train.Saver()

env = TorcsEnv(vision=False, throttle=True, gear_change=False)

### ----------------------------------------------------------------------- ###

file_path = 'Best/bestlaptime.csv'
r_w = rw.RW(file_path)
best_lap_time = r_w.read_numpy_array_from_csv()
print(best_lap_time)
w_csv = w_Out.OW(csv_path = 'OutputCsv/output.csv')
w_total_csv = w_Out.OW(csv_path = 'OutputCsv/output_total.csv')

### ----------------------------------------------------------------------- ###
#actor_losses = []
#critic_losses = []
steps = []
#actor_loss, crit_loss = [0.0 , 0.0]

### ----------------------------------------------------------------------- ###

if (train_test == 0 and irestart == 0):
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, "ckpt/model")  


for ep in range(iter_num, EP_MAX):

    print("-"*50)
    print("episode: ", ep)

    if np.mod(ep, relaunch_le) == 0:
        ob = env.reset(relaunch=True)   #relaunch TORCS every N episode because of the memory leak error
    else:
        ob = env.reset()

    s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0

    for t in range(EP_LEN):    # in one episode
       
        a = ppo.choose_action(s)

        
        a[0] = np.clip(a[0],-1.0,1.0)
        a[1] = np.clip(a[1],0.0,1.0)
        a[2] = np.clip(a[2],0.0,1.0)  

        #print("a: ", a)


        ob, r, done, _, end_type = env.step(a)
        
        ### LAST LAP TIME ### CHECK HERE
        if ob.lastLapTime > 0:
            print("lap time is : ",ob.lastLapTime)
            if (ob.lastLapTime < best_lap_time) and (train_test==0):
                best_lap_time = ob.lastLapTime
                r_w.write_numpy_array_to_csv(best_lap_time)
                saver.save(sess, "Best/model")
                print("Best Lap Time is updated.")
                print("saving Best model")
        
        s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))  


        if (train_test == 0):
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)    

        s = s_
        ep_r += r

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
                
        print("="*100)
        print("--- Episode : {:<4}\tActions ".format(ep)+ np.array2string(a, formatter={'float_kind': '{0:.3f}'.format})+"\tReward : {:8.4f}".format(ep_r)+" ---")
        print("="*100)
        #print("Actor Loss: "+str(actor_loss)+ "\tCrit Loss: "+ str(crit_loss))
        
        #----------------------------------------------------------------------------------------------------------------
        # Saving outputs to csv file
        #print("saving csv")
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
    # EDIT HERE AFTER
    #output_total_csv = np.hstack((ep, t, end_type, ob.distRaced, ob.distFromStart, ob.curLapTime, ob.lastLapTime, ep_r, actor_loss, crit_loss))
    output_total_csv = np.hstack((ep, t, end_type, ob.distRaced, ob.distFromStart, ob.curLapTime, ob.lastLapTime, ep_r))
    w_total_csv.append_numpy_array_to_csv(np.matrix(output_total_csv))
    ### ----------------------------------------------------------------------------- ###
    
    

    if (train_test == 0):
        with open("performance.txt", "a") as myfile:
            myfile.write(str(ep) + " " + str(t) + " " + str(round(ep_r,4)) + "\n")

    if np.mod(ep, 100) == 99:
        if (train_test ==0):
            file_name = 'Models/'+str(ep+1)
            if os.path.isdir(file_name) ==False:
                os.mkdir(file_name)
            model_name = file_name+'/model'
            print("saving model")
            saver.save(sess, model_name)


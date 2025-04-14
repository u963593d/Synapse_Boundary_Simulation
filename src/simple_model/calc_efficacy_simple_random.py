import os
import sys
import numpy as np
import subprocess


##args
#calc_efficacy_simple_random.py learning_rule fitting_curve num_of_neurons time_block(ms) step_size(ms) save_folder_name firing_pattern_for_wake firing_pattern_for_sleep sleep_or_wake(s:sleep, w:wake, sw:sleep and wake) seed block_num_for_GPU option
args = sys.argv

#python3 calc_efficacy_simple_random.py Hebbian a0.7t50 10 60000 0.05 synv_by_random ISIM_40_0.05 ISIM1.5_UPM2.7_45_0.05 s 6 10
Hz="0.10"  #firing rate being searched for the number of items already calculated

lr = args[1]
lr_p = args[2]

NE = args[3]
Tp = args[4]
dt = args[5]
syn_fol = args[6] 

dsb_w = args[7] #"ISI_40_0.05"
dsb_s= args[8]

sw= args[9]
savedir = os.getcwd() +"/"
seed = args[10]
blockdim = args[11] 


os.makedirs(savedir+syn_fol+"/", exist_ok=True)
fols  = os.listdir(savedir+syn_fol+"/")

if sw =="s":
    for fol in fols:
        if dsb_s in fol:
            dsb_s = fol
            break
    print("dsb_s", dsb_s) 
elif sw=="w":
    for fol in fols:
        if dsb_w in fol:
            dsb_w = fol
            break
    print("dsb_w", dsb_w) 
    
elif sw=="sw":
    for fol in fols:
        if dsb_s in fol:
            dsb_s = fol
            break
    print("dsb_s", dsb_s) 
    
    for fol in fols:
        if dsb_w in fol:
            dsb_w = fol
            break
    print("dsb_w", dsb_w) 



if sw =="s":
    fori = syn_fol+ "/"+dsb_s+"/"+lr_p+"/N_"+str(NE)
    f = savedir + fori + "/"+lr+"_"+ "sleep"+Hz+".csv"
elif sw=="w":
    fori = syn_fol+ "/"+dsb_w+"/"+lr_p+"/N_"+str(NE)
    f = savedir + fori + "/"+lr+"_"+ "wake"+Hz+".csv"
elif sw=="sw":
    fori = syn_fol+ "/"+dsb_s+"/"+lr_p+"/N_"+str(NE)
    f = savedir + fori + "/"+lr+"_"+ "sleep"+Hz+".csv"

if os.path.exists(f):
    syn_v= np.genfromtxt(f, delimiter=",", dtype=np.float64)
    last = int(syn_v[len(syn_v)-1][0])
    print("last_num", last)
else:
    last=-1

make_f_str = "python3 make_effi_sh_random.py {} {} {} {} {} {}".format(lr, lr_p, last+1, None, "GPU", "r")

print(make_f_str)
subprocess.run([make_f_str], shell=True)

if sw =="s":
    GPU_str = "sh {}_{}_GPU_r_s_off{}.sh {} {} {} {} {} {} {} {} {}".format(lr, lr_p, int(last+1), NE, Tp, dt, syn_fol, dsb_w, dsb_s, lr_p, seed, blockdim)
    # subprocess.run([GPU_str], shell=True)
elif sw=="w":
    GPU_str = "sh {}_{}_GPU_r_w_off{}.sh {} {} {} {} {} {} {} {} {}".format(lr, lr_p, int(last+1), NE, Tp, dt, syn_fol, dsb_w, dsb_s, lr_p, seed, blockdim)
    # subprocess.run([GPU_str], shell=True)
elif sw=="sw":
    GPU_str = "sh {}_{}_effi_GPU_r_off{}.sh {} {} {} {} {} {} {} {} {}".format(lr, lr_p, int(last+1), NE, Tp, dt, syn_fol, dsb_w, dsb_s, lr_p, seed, blockdim)
    # subprocess.run([GPU_str], shell=True)


print(GPU_str)
subprocess.run([GPU_str], shell=True)






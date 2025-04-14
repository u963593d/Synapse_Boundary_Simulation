import os
import sys
import numpy as np
import subprocess


##args
#calc_efficacy_simple.py learning_rule fitting_curve num_of_neurons time_block(ms) step_size(ms) save_folder_name firing_pattern_for_wake firing_pattern_for_sleep sleep_or_wake(s:sleep, w:wake, sw:sleep and wake) seed block_num_for_GPU option
args = sys.argv  
Hz="0.10"  #firing rate being searched for the number of items already calculated

##examples
#standard N=10, a0.7t50 (Fig. 1)
#python3 calc_efficacy_simple.py STDP a0.7t50 10 60000 0.05 synv_by ISIM_40_0.05 ISIM1.5_UPM2.7 s 6 10 None

#N=96 (Fig. 2)
#python3 calc_efficacy_simple.py STDP a0.7t50 96 60000 0.05 synv_by ISIM_40_0.05 ISIM1.5_UPM2.7 s 6 32 None

#another fittinh curve (Fig. 2 and Fig. S3)
#python3 calc_efficacy_simple.py STDP a0.9t50 10 60000 0.05 synv_by ISIM_40_0.05 ISIM1.5_UPM2.7 w 6 10 None

#STDP sleep chaging ISIM (Fig. 2)
#python3 calc_efficacy_simple.py STDP a0.7t50 10 60000 0.05 synv_by ISIM_40_0.05 DOWNM3.0_UPM2.7 s 6 10 None

#STDP sleep chaging DOWNM (Fig. 2)
#python3 calc_efficacy_simple.py STDP a0.7t50 10 60000 0.05 synv_by ISIM_40_0.05 ISIM1.0_UPM2.1 s 6 10 None
#python3 calc_efficacy_simple.py STDP a0.7t50 10 60000 0.05 synv_by ISIM_40_0.05 ISIM2.25_UPM2.7 s 6 10 to3hz   #(0.1~3Hz in ISIM2.0, ISIM2.25)

#high hz (Fig. S2)
#python3 calc_efficacy_simple.py STDP a0.7t50 10 60000 0.05 synv_by ISIM_40_0.05 DOWNM2.7_UPM2.7 sw 6 10 high

#mean firing rates in the up state of sleep-like firing pattern = mean firing  rates in wake-like firing pattern (Fig. 5)
#python3 calc_efficacy_simple.py Hebbian a0.7t50 10 60000 0.05 synv_by_mf ISIM_40_0.05 DOWNM3.0_UPM2.7_157_0.05 s 6 10 mf
#python3 calc_efficacy_simple.py Hebbian a0.7t50 10 60000 0.05 synv_by_mf ISIM_40_0.05 DOWNM2.7_UPM2.7 w 6 10 mf

#difference of membrane potential between the state of sleep and wake (FIg. S)
#python3 calc_efficacy_simple.py STDP a0.7t50 10 60000 0.05 synv_by diff5_40_0.05 diff5_40_0.05 s 6 10 None



lr = args[1]
lr_p = args[2]

NE = args[3]
Tp = args[4]
dt = args[5]
syn_fol = args[6] 

dsb_w = args[7] #"ISIM_40_0.05", 
dsb_s= args[8]
sw= args[9]
savedir = os.getcwd() +"/"
seed = args[10]
blockdim = args[11] 
op = args[12] 

start_f = 0

# if os.path.exists(savedir+syn_fol+"/"):
os.makedirs(savedir+syn_fol+"/", exist_ok=True)
fols  = os.listdir(savedir+syn_fol+"/")

if sw =="s":
    for fol in fols:
        if dsb_s in fol:
            dsb_s = fol
            start_f =1
            break
    print("dsb_s", dsb_s) 
elif sw=="w":
    for fol in fols:
        if dsb_w in fol:
            dsb_w = fol
            start_f =1
            break
    print("dsb_w", dsb_w) 
    
elif sw=="sw":
    for fol in fols:
        if dsb_s in fol:
            dsb_s = fol
            start_f =1
            break
    print("dsb_s", dsb_s) 
    
for fol in fols:
    if dsb_w in fol:
        dsb_w = fol
        start_f =1
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

if last!=999:

    if op =="None":
        make_f_str = "python3 make_effi_sh.py {} {} {} {} {} {}".format(lr, lr_p, last+1, None, "GPU", "c")
    else:
        make_f_str = "python3 make_effi_sh_op.py {} {} {} {} {} {} {}".format(lr, lr_p, last+1, None, "GPU", "c", op)

    print(make_f_str)
    subprocess.run([make_f_str], shell=True)


    if op =="None":
        if sw =="s":
            GPU_str = "sh {}_{}_GPU_s_off{}.sh {} {} {} {} {} {} {} {} {}".format(lr, lr_p, int(last+1), NE, Tp, dt, syn_fol, dsb_w, dsb_s, lr_p, seed, blockdim)
        
        elif sw=="w":
            GPU_str = "sh {}_{}_GPU_w_off{}.sh {} {} {} {} {} {} {} {} {}".format(lr, lr_p, int(last+1), NE, Tp, dt, syn_fol, dsb_w, dsb_s, lr_p, seed, blockdim)
            
        elif sw=="sw":
            GPU_str = "sh {}_{}_effi_GPU_off{}.sh {} {} {} {} {} {} {} {} {}".format(lr, lr_p, int(last+1), NE, Tp, dt, syn_fol, dsb_w, dsb_s, lr_p, seed, blockdim)
        

    else:
        if sw =="s":
            GPU_str = "sh {}_{}_GPU_s_off{}_{}.sh {} {} {} {} {} {} {} {} {}".format(lr, lr_p, int(last+1), op,NE, Tp, dt, syn_fol, dsb_w, dsb_s, lr_p, seed, blockdim)
            
        elif sw=="w":
            GPU_str = "sh {}_{}_GPU_w_off{}_{}.sh {} {} {} {} {} {} {} {} {}".format(lr, lr_p, int(last+1), op,NE, Tp, dt, syn_fol, dsb_w, dsb_s, lr_p, seed, blockdim)
            
        elif sw=="sw":
            GPU_str = "sh {}_{}_effi_GPU_off{}_{}.sh {} {} {} {} {} {} {} {} {}".format(lr, lr_p, int(last+1), op,NE, Tp, dt, syn_fol, dsb_w, dsb_s, lr_p, seed, blockdim)
            

    print(GPU_str)
    subprocess.run([GPU_str], shell=True)
else:
    print("already end")
   
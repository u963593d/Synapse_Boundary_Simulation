import os 
import sys
from scipy.signal import find_peaks

import numpy as np
import gc

import numpy as np



#python3 get_c_r.py +" " +args 
#python3 get_c_r.py 64 20 1.0 1.5 0.01 0.01 0.01 0.01 0 1.0 800 400 400 0.0002 2.0 2.0 4 r True 23 SANAG_NaK5_log

offset =1000#250 #ms
  


def c_r_from_wake(ex, ex_in, N, tbs, drc_syn, cr_drc, offset, dt):
    
    cpre_peaks = np.empty(0)
    cpost_peaks = np.empty(0)
    offset = 250
    for post_i in range(N):
        if post_i !=0:
            continue
        for s in range(N):
                cpre_file = drc_syn+  "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"cpre"+str(s)+"-"+str(post_i)+"_"+str(0)+".bin"
                print(cpre_file)
                if os.path.exists(cpre_file):
                    print(cpre_file)
                
                    for tb in range(tbs):
    #                     print("tb", tb)
                        cpre_file =drc_syn+  "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"cpre"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
            #                    
                        fcpre= open(cpre_file, "rb")
                        rectype = np.dtype(np.float64)
                        cpre_tb = np.fromfile(fcpre, dtype=rectype).astype("float32")
                        if tb==0:
                            cpre=cpre_tb
                        else:
                            cpre=np.append(cpre, cpre_tb).astype("float32")
                        fcpre.close()
                    # if np.isnan(cpre):
                    #     print("cpre; nan")
                        
                    peak_i = find_peaks(cpre[int(offset/dt):], height=0)[0]
                    print("peak_i", peak_i)
                    m_peaks = np.mean(np.take(cpre[int(offset/dt):], peak_i))
                    cpre_peaks = np.append(cpre_peaks, m_peaks)
                    print("peaks", np.take(cpre[int(offset/dt):], peak_i))

                    for tb in range(tbs):
    #                     print("tb", tb)
                        cpost_file =drc_syn+  "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"cpost"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
                        fcpost= open(cpost_file, "rb")
                        rectype = np.dtype(np.float64)
                        cpost_tb = np.fromfile(fcpost, dtype=rectype).astype("float32")
                        if tb==0:
                            cpost=cpost_tb
                        else:
                            cpost=np.append(cpost, cpost_tb).astype("float32")
                        fcpost.close()
                        
                    peak_i = find_peaks(cpost[int(offset/dt):], height=0)[0]
    #                 print("peak_i", peak_i)
                    m_peaks = np.mean(np.take(cpost[int(offset/dt):], peak_i))
                    cpost_peaks = np.append(cpost_peaks, m_peaks)

                    del cpre
                    del cpre_tb
                    del cpost
                    del cpost_tb
                    gc.collect()
                    
    cpre_m = np.mean(cpre_peaks)
    cpost_m = np.mean(cpost_peaks)
    print("mean cpre", cpre_m)
    print("mean cpost", cpost_m)
    # cpre_r = 0.7/cpre_m
    # cpost_r = 1.4/cpost_m
    cpre_r_pre = 0.7/cpre_m
    cpost_r_pre = 1.4/cpost_m
    
    cpre_r=1.0
    cpost_r = cpost_r_pre/cpre_r_pre
    
    print("cpre_r", cpre_r)
    print("cpost_r", cpost_r)
    
 
    path = cr_drc + "c_r_w.txt"
    try:
        os.makedirs(cr_drc)
    except FileExistsError:
        print("{} is already exist".format(cr_drc))  
    try:
        with open(path, mode = "w")  as f:
            f.write(str(cpre_m)+"\n")
            f.write(str(cpost_m)+"\n")
            f.write(str(cpre_r)+"\n")
            f.write(str(cpost_r)+"\n")
        return cpre_r, cpost_r
    except:
        print("not found: {}".format(path))

args = sys.argv
vc = 1
g_nmdar = float(args[vc])
vc+=1
print("g_nmdar", g_nmdar)
g_cav=float(args[vc])
vc+=1
cpre_r = float(args[vc])
vc+=1
cpost_r = float(args[vc])
vc+=1
print("cpost_r", cpost_r)
tau_ca_pre = float(args[vc])
vc+=1
tau_ca_post = float(args[vc])
vc+=1

NE = int(args[vc]) #[16, 32, 64, 128]
vc+=1
print("NE", NE)
Ip = float(args[vc])
vc+=1
con_M = float(args[vc])
vc+=1
con_M_in= float(args[vc])
vc+=1
con_ex_ex = float(args[vc])  #ex to ex  #conSD
vc+=1

con_ex_in = float(args[vc])  #ex to in 
vc+=1
con_in_ex = float(args[vc])    #conSDin
vc+=1
con_in_in = float(args[vc]) 
vc+=1

T =  int(args[vc])
vc+=1
print("T", T)
Tp=  int(args[vc])
vc+=1
Tc= float(args[vc])
vc+=1
dt =  float(args[vc]) 
vc+=1
print("dt", dt)
exp_ex = float(args[vc])
vc+=1
exp_in = float(args[vc])
vc+=1
seed =  int(args[vc])
vc+=1
init = (args[vc])
vc+=1
date_i = (args[vc]) #[16, 32, 64, 128]
vc+=1
model_name = args[vc]
vc+=1

con_log = (args[vc])
vc+=1

print("model_name", model_name)



# ex_diff = (exp_in - exp_ex)
NI = int(NE*Ip/(100-Ip))
tbs=int(Tc/Tp)

# nowdir = os.getcwd()

# savedir ="/mnt/SSD1/net_search/" 

# try:
#     os.makedirs(par_f_dir)
# except FileExistsError:
#     print("{} is already exist".format(par_f_dir)) 

if con_log=="False":
    drc ="./con_cu_i/{}/param_{}/cprer{}_cpostr{}_taupre{}_taupost{}/NE{}_NI{}/con_th{}_{}_{}_{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/syn/".format(model_name, date_i, cpre_r, cpost_r, tau_ca_pre, tau_ca_post,NE, NI, con_ex_ex,con_ex_in,con_in_ex,con_in_in, seed,init,T,Tp,dt)
elif con_log=="True":
    drc ="./con_cu_i/{}/param_{}/cprer{}_cpostr{}_taupre{}_taupost{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/syn/".format(model_name, date_i,  cpre_r, cpost_r , tau_ca_pre, tau_ca_post,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in, seed,init,T,Tp,dt)

if con_log=="False":
    cr_drc ="./con_cu_i/{}/param_{}/cprer{}_cpostr{}_taupre{}_taupost{}/NE{}_NI{}/con_th{}_{}_{}_{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/cr_ex{}_exin{}/".format(model_name, date_i, cpre_r, cpost_r, tau_ca_pre, tau_ca_post,NE, NI, con_ex_ex,con_ex_in,con_in_ex,con_in_in, seed,init,T,Tp,dt,exp_ex,exp_in)
elif con_log=="True":
    cr_drc ="./con_cu_i/{}/param_{}/cprer{}_cpostr{}_taupre{}_taupost{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/cr_ex{}_exin{}/".format(model_name, date_i,  cpre_r, cpost_r , tau_ca_pre, tau_ca_post,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in, seed,init,T,Tp,dt,exp_ex,exp_in)

print(cr_drc)
# tau_ca_pre= float(args[vc])
# vc+=1
# tau_ca_post= float(args[vc])
# vc+=1
# cpre_r= float(args[vc])
# vc+=1
# cpost_r= float(args[vc])
# vc+=1

if __name__=="__main__":
    cpre_r, cpost_r = c_r_from_wake(exp_ex, exp_in, NE, tbs, drc, cr_drc, offset, dt)
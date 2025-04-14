

import os
import subprocess
import sys
import pandas as pd
from multiprocessing import Pool



#calculating time change in synaptic efficacy in network models under synaptic learning rules

#python3 calc_network_lr_sw.py learning_rule fitting_curve CPU_core
#python3 calc_network_lr_sw.py STDP a0.7t50th0.6 1


nowdir = os.getcwd()+"/"
savedir = nowdir + "cc_dir/"  #results of bifurcation analysis in network models
bifurcation_net_checked_file = "net_lr_data_pre_ex.xlsx" #sheet for collections of parameter sets that will be calculated


args = sys.argv
lr_p = args[2]
lr = args[1]

cores = int(args[3])


param_n="ori"    #log(changing SD for the lognormal distribution of the number of synapses), #syn(changing mean for the lognormal distribution of the number of synapses )
op = "_net_bifurcation"  #without IN-IN connection
model_op=op
model_pre ="AN"  #Averaged Neuron model
model_op_effi = "_net_effi"


#connection parameters
NE = 64  # number of excitatory neurons
Ip = 20  # ratio of inhibitory neurons 
NI = int(NE*Ip/(100-Ip)) #number of inhibitory neurons
con_log = True  #True -> lognormal connections, False -> random connections
con_M = 1.0  # mean for the lognormal distribution of the number of synapses per excitatory neuron
con_M_in =  1.0 # mean for the lognormal distribution of the number of synapses per inhibitory neuron
log=0.01
con_ex_ex = log # SD for the lognormal distribution of the number of synapses in Ex-Ex connections
con_ex_in = log  # SD for the lognormal distribution of the number of synapses in Ex-In connections
con_in_in = log  # SD for the lognormal distribution of the number of synapses in In-In connections
con_in_ex=log  # SD for the lognormal distribution of the number of synapses in In-Ex connections

T = 5000  # ms   simulation time
Tp=1000
dt = 0.04

cpre_r = 1.0  #initial coefficient for calcium influx by pre-synaptic spike
cpost_r =1.0  #initial coefficient for calcium influx by post-synaptic spike
init ="r" #r->random initial value   , c->constant initial value
seed=4

smin_ampa = 0.5  #lower limit of coefficient for AMPAR conductance
smax_ampa = 1.5 #upper limit of coefficient for AMPAR conductance
blockdim_x =NE+NI   #number of GPU threads


# read files for information of network model
df_net = pd.read_excel(nowdir + bifurcation_net_checked_file , engine="openpyxl")

def multi_lrsw(cores, lr, lr_p, df_net, model_pre, model_op,param_n, NE,Ip,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_in,con_in_ex, T, Tp, seed, init):
    args = []
    for i in range(len(df_net)):
    #     if i <60:
    #         continue
        date_i = df_net.iloc[i,0]

        ex_lr =  df_net.iloc[i,1]
        ex_w = df_net.iloc[i,2]
        ex_s = df_net.iloc[i,3]
#       
        df_i = df_net.iloc[i,:]
        
        bifur ="_"+str(df_i["bifur"])
#        
        print("bifur", bifur)
  
        model_name = model_pre +model_op + bifur
        model_name_lr = model_pre +model_op_effi + bifur + "_"+lr + "_w"
       

        
        cc_x =  savedir  + "{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}_cc.xlsx".format(model_name, date_i, NE,NI,con_M,con_ex_ex,con_M_in,con_in_in,T)
        
        res_dir = "./con_cu_i/{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/sminampa{}_smaxampa{}/{}_{}/{}/".format(model_name_lr, NE, NI,con_M, log,con_M_in, log,smin_ampa,smax_ampa,lr,lr_p,date_i)
        
        if os.path.exists(res_dir + "sw_end.csv"):
            print("{} is already analyzed".format(date_i))
            continue

        if not os.path.exists(cc_x):
            print("{} is not found".format(cc_x))
            continue

        print("{} will be analyzed".format(date_i))

        
        args.append((i, lr, lr_p, df_net,model_name, param_n,  date_i, NE,Ip,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_in,con_in_ex, T, Tp, seed, init, ex_lr, ex_w, ex_s, bifur))
    
    print(f'simulate network sw: using {cores} cores')
    with Pool(processes=cores) as pool:
        pool.map(lrsw_gpu, args)
        
def lrsw_gpu(args):
    i, lr, lr_p, df_net, model_name, param_n,  date_i, NE,Ip,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_in,con_in_ex,  T, Tp, seed, init, ex_lr_ori, ex_w, ex_s, bifur = args
    
    lr_dir ="./lr_params_con/{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}/".format(model_name, date_i, NE,NI,con_M,con_ex_ex,con_M_in,con_in_in)
    net_lr_ok =False

    if os.path.exists(lr_dir):
        exs_lr =  os.listdir(lr_dir)
        print("exs_lr ", exs_lr)
        for ex_lr in exs_lr:
            lr_dir2 = lr_dir +str(ex_lr) + "/"
#                 print(lr_dir2)
            if os.path.exists(lr_dir2+lr+"_"+lr_p +".xlsx"):
                net_lr_ok =True
                print("net_lr_ok True")
                break
      
    out_lr_search = "python3 net_lr_search.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(lr, lr_p, model_name, date_i, NE,Ip,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_in,con_in_ex, T, Tp,  seed, init, ex_lr_ori, bifur, param_n)


    out_net_lr = "python3 exc_calc_sw.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(lr, lr_p, model_name, date_i, NE,Ip,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_in,con_in_ex, T, Tp, seed, init, ex_w, ex_s, bifur)
    print(out_net_lr)

    out_check_fre = "python3 check_freq_effi.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(lr, lr_p, model_name, date_i, NE,Ip,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_in,con_in_ex, T, Tp, seed, init, ex_w, ex_s, bifur)
    print(out_check_fre)
    
    if net_lr_ok == False:
#         print(lr_dir2+lr+"_"+lr_p +".xlsx")
        print(out_lr_search)
        subprocess.run([out_lr_search], shell=True)
        
        

    subprocess.run([out_net_lr], shell=True)
    subprocess.run([out_check_fre], shell=True) 
#     print("doing: {}".format(sh_f))



if __name__ == "__main__":
    multi_lrsw(cores, lr, lr_p, df_net,  model_pre,  op, param_n,  NE,Ip,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_in,con_in_ex, T, Tp, seed, init)

   
        
        
        




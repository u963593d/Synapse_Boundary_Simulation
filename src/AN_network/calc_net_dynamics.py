 
import os
import subprocess
import sys
import pandas as pd
import traceback

#python3 calc_net_dynamics.py type_of_bifurcation(nmdar, cav or pre) molecules_updating_conductance model_index
#python3 calc_net_dynamics.py pre aup 2023_12_6_7_10

args = sys.argv

 
bifur = "_"+args[1]    #types of bifurcation model
op = "_net_"+args[2]       #"aup" -> update conductances by the second phosphorylated states of kinases, "rup" -> update conductances by the initial phosphorylated states of kinases

model_op= op+""  # postfix for model name
op_param="_net_bifurcation"  # postfix for parameter files for conductances
op_cr_name = op + "_crw"
model_pre = "AN"   #prefix model name

pre_simu_dir = "cadyn_dir_crw" #directory for the results of preliminary simulations
 
date_i = args[3]  #index for network model


do_cargs=True   #doing the GPU network calculations
get_c_r_w=True   #doing preliminary simulations to obtain scaling parameters and normalized thresholds
rho_sw_com = True
th_complement = True  #doing normalization of thresholds
cv_th = 1.0   #sleep score threshold for sleep states


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
con_in_ex = log  # SD for the lognormal distribution of the number of synapses in In-Ex connections


#synaptic learning rules
lr ="STDP"
lr_p="a0.7t50th0.6"
cpre_r = 1.0  #initial coefficient for calcium influx by pre-synaptic spike
cpost_r =1.0  #initial coefficient for calcium influx by post-synaptic spike

rho_ini =0.5  #initial synaptic efficacy
smin_ampa = 0.5  #lower limit of coefficient for AMPAR conductance
smax_ampa = 1.5 #upper limit of coefficient for AMPAR conductance

#sleep-wake dynamics
alphas = [1100000.0]   #coefficient for Ca2+ activation
betas = [0.25]    #coefficient for Ca2+ inhibition
ths= [0.0]        #noise 
sig_ns=[0.0]     #noise
ws = [100.0]     #coefficient for auto activation
bs = [80.0]     #coefficient for inhibition of second states against first states
Is = [-18.0]     #current 

adps = [0.65] 
bdps = [80.0] 
adp2s =[0.0]
bdp2s = [0.25]
tau_camks =[2500]   #time constant for the initial phosphorylated states of kinases
tau_ks = [10000]   #time constant for the second phosphorylated states of kinases

ex= -1.8          # initial conductance = original conductance X10^ex,   power of receptor or channel conductance
ex_w=ex
ex_in_w=ex_w
ex_in=ex_in_w
ex=ex_w
max_cons = [4.0]   #max rate for receptor or channel conductance

UA=1.0  #Upper asymptote for process S
LA=0.0   #Lower asymptote for process S
ti = 20000  # time constant of the increasing exponential saturating function with an upper asymptote
td = 20000   # time constant   of the decreasing exponential function with an lower asymptote

r_ini = 0.9 # initial ratio of the initial phosphorylated states of kinases
a_ini = 0.1   # initial ratio of the second phosphorylated states of kinases
z_ini = 0.0  #initial value for noise

#others
T = 300000  #ms, Total time for simulation
Tw = 150000  #ms, time for preliminary simulation
Tp = 5000  #ms,  time block   

tbs=int(T/Tp)
blockdim_x = int(NE+NI)            #number of GPU threads        
init = "r"  #r->random initial value   , c->constant initial value
seed = 6

nav=1
kva =1
kvsi=1
kir=1
nmdar=1
gabar=1


postfix = "NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}".format(NE,NI,con_M,con_ex_ex,con_M_in,con_in_in,2500)

#read parameter files for channel and receptor conductances
param_cond_dir ="./net_col/"
model_param = model_pre  +op_param + bifur
df_params = pd.read_excel(param_cond_dir+model_param+"_"+postfix+"_bifur_params.xlsx", engine="openpyxl")
df_params_i=df_params[df_params["order"]==date_i]
df_param = df_params_i.iloc[0,1:14]
print("pars", df_param)
pars= df_param.to_dict()


#read parameter files for synaptic learning rules
lr_dir ="./lr_params_con/{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}/".format(model_param, date_i, NE,NI,con_M,con_ex_ex,con_M_in,con_in_in) 
net_lr_ok =False
exs_lr =  os.listdir(lr_dir)
for ex_lr in exs_lr:
    if "xlsx" in ex_lr:
        continue
    lr_dir2 = lr_dir +str(ex_lr) + "/"
    if os.path.exists(lr_dir2+lr+"_"+lr_p +".xlsx"):
        net_lr_ok =True
        pars_lr_ori=pd.read_excel(lr_dir2+lr+"_"+lr_p +".xlsx", engine="openpyxl", index_col=0)
        break
    


print("pars_lr_ori", pars_lr_ori)
pars_lr_ori=(pars_lr_ori.T).to_dict()
pars_lr_ori = pars_lr_ori[0]
print("pars_lr_ori", pars_lr_ori)


gp = round(pars_lr_ori['gp'],3)
gd =round(pars_lr_ori['gd'],3)
sig = round(pars_lr_ori["sig"],4)
tau_ca_pre = round(pars_lr_ori['tau_ca_pre'],3)
tau_ca_post =round(pars_lr_ori['tau_ca_post'],3)
pars_lr_ori['tau_ca_pre']=tau_ca_pre
pars_lr_ori['tau_ca_post']=tau_ca_post
pars_lr_ori['gp']= gp
pars_lr_ori['gd']=gd
pars_lr_ori['sig']=sig
tau_ss = [pars_lr_ori["tau_s"]] #

pars_lr2 = {}
pars_lr2["th_p"]=  pars_lr_ori["th_p"]
pars_lr2["th_d"]=  pars_lr_ori["th_d"]
pars_lr2["gp"]=  pars_lr_ori["gp"]
pars_lr2["gd"]=  pars_lr_ori["gd"]
pars_lr2["tau_ca_pre"] = pars_lr_ori["tau_ca_pre"]
pars_lr2["tau_ca_post"] = pars_lr_ori["tau_ca_post"]
pars_lr2["sig"] = pars_lr_ori["sig"]
pars_lr2["tau_s"] = pars_lr_ori["tau_s"]


if nav==1 and kva ==1 and kvsi==1 and kir==1 and gabar==1: #NaK1
    keys = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_ampar','g_nmdar', 'g_gabar', 't_ca']
    ext = "AN"
elif nav==0 and kva ==1 and kvsi==1 and kir==1 and gabar==1: #K1
    keys = ['g_leak', 'g_kvhh', 'g_kva', 'g_kvsi', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K1"
elif nav==1 and kva ==1 and kvsi==1 and kir==0 and gabar==1: #NaK3
    keys = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 'g_cav', 'g_kca', 'g_nap',  'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK3"
elif nav==0 and kva ==1 and kvsi==1 and kir==0 and gabar==1: #K3
    keys = ['g_leak',  'g_kvhh', 'g_kva', 'g_kvsi', 'g_cav', 'g_kca', 'g_nap', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K3"
elif nav==1 and kva ==0 and kvsi==1 and kir==0 and gabar==1: #NaK6
    keys = ['g_leak', 'g_nav', 'g_kvhh',  'g_kvsi', 'g_cav', 'g_kca', 'g_nap',  'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK6"
elif nav==0 and kva ==0 and kvsi==1 and kir==0 and gabar==1: #K6
    keys = ['g_leak',  'g_kvhh',  'g_kvsi', 'g_cav', 'g_kca', 'g_nap', 'g_ampar', 'g_gabar', 't_ca']
elif nav==1 and kva ==1 and kvsi==0 and kir==0 and gabar==1: #NaK5
    keys = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva','g_cav', 'g_kca', 'g_nap', 'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK5"
elif nav==0 and kva ==1 and kvsi==0 and kir==0 and gabar==1: #K5
    keys = ['g_leak',  'g_kvhh', 'g_kva', 'g_cav', 'g_kca', 'g_nap', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K5"
elif nav==1 and kva ==0 and kvsi==0 and kir==1 and gabar==1: #NaK7
    keys = ['g_leak', 'g_nav', 'g_kvhh', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK7"
elif nav==0 and kva ==0 and kvsi==0 and kir==1 and gabar==1: #K7
    keys = ['g_leak', 'g_kvhh', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K7"
elif nav==1 and kva ==0 and kvsi==1 and kir==1 and gabar==1: #NaK4
    keys = ['g_leak', 'g_nav', 'g_kvhh',  'g_kvsi', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK4"
elif nav==0 and kva ==0 and kvsi==1 and kir==1 and gabar==1: #K4
    keys = ['g_leak',  'g_kvhh',  'g_kvsi', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K4"
elif nav==1 and kva ==1 and kvsi==0 and kir==1 and gabar==1: #NaK2
    keys = ['g_leak', 'g_nav', 'g_kvhh',  'g_kva', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK2"
elif nav==0 and kva ==1 and kvsi==0 and kir==1 and gabar==1: #K2
    keys = ['g_leak', 'g_kvhh',  'g_kva', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K2"
    
model_pre = ext

model_name = model_pre + model_op + bifur
model_name_cr =  model_pre +  op_cr_name + bifur
out = "./"+model_pre + op + bifur
out_cr = "./"+model_pre+ op + bifur

sh_f = "net_search_"+ model_name + "_"+date_i+"_lr"+op+bifur+".sh"

nowdir = os.getcwd()
savedir =nowdir+"/" 

mode = "w" 
ftxt = savedir + sh_f

if mode == "w":
    f0 = open(ftxt, "w")
    prefix = "#!/usr/bin/bash"
    f0.write(prefix+'\n')
else:
    f0 = open(ftxt, "a")

params_str = ""
for k in keys:
    params_str += str(pars[k])
    params_str+= " "

      
print("tau_ca_pre", pars_lr_ori["tau_ca_pre"])
                    

for alpha in alphas:
    for beta in betas:
        for adp in adps:
            for bdp in bdps:
                for adp2 in adp2s:
                    for bdp2 in bdp2s:
                        for tau_camk in tau_camks:
                            for th in ths:
                                for sig_n in sig_ns:
                                    for tau_k in tau_ks:
                                        for w in ws:
                                            for b in bs:
                                                for I in Is:
                                                    for max_con in max_cons:
                                                    
                                                   
                                                        if get_c_r_w==True:
                                                            pars_lr=pars_lr2.copy()
                                                            cpre_r=1.0
                                                            cpost_r=1.0
                                                            pars_lr_pre = pars_lr.copy()
                                                            pars_lr_pre["tau_s"]=0.0                 
                                                            params_lr_str_pre = str(1.0) + " " + str(1.0) +" " 
                                                            for k, v in pars_lr_pre.items():
                                                                params_lr_str_pre += str(pars_lr_pre[k])
                                                                params_lr_str_pre+= " "       
                                                            if con_log==True:
                                                                cr_drc ="./"+pre_simu_dir+"/{}/param_{}/NE{}_NI{}/conM{}_SD{}_conMin{}_inSD{}/Spini{}_Kini{}_zini{}/exp{}_{}_{}_{}_{}_{}/taucamk{}_tauk{}/w{}_b{}_I{}_th{}_sign{}/T_{}/ex{}_exin{}/".format(model_name_cr, date_i, NE,NI,con_M,con_ex_ex,con_M_in, con_in_in,r_ini, a_ini, z_ini, alpha,beta,adp,bdp,adp2,bdp2, tau_camk, tau_k,w,b,I,th,sig_n,Tw,ex,ex)
                                                            print("cr_drc", cr_drc)
                                                            fline_cr = out_cr +" "+ params_str+str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+" " +params_lr_str_pre +" "+str(r_ini)+" "+str(a_ini)+" " +str(z_ini)+" " +str(alpha)+" "+str(beta)+" "+str(adp)+" "+str(bdp)+" " +str(adp2)+" "+str(bdp2)+" "+ str(tau_camk) +" "+ str(tau_k) +" " +str(w)+ " "+str(b)+" "+str(I)+" "+str(th)+ " "+str(sig_n)+" "+str(max_con) +" "+str(rho_ini)+" " +str(smin_ampa) +" " +str(smax_ampa) +" " +str(Tw) +" " +str(Tp) + " "+str(ex)+" "+str(ex_in)+" "+str(seed) + " " +init +" "+  date_i+  " "+ model_name_cr + " " +str(blockdim_x)
          
                                                            
                                                            py_cr = "python3 analysis_net_dynamics_crw.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(model_name_cr, date_i,cpre_r,cpost_r,pars_lr["th_p"],pars_lr["th_d"],pars_lr["gp"],pars_lr["gd"], pars_lr["tau_ca_pre"],pars_lr["tau_ca_post"],pars_lr["sig"],pars_lr["tau_s"],  r_ini, a_ini, z_ini, alpha, beta,  adp,bdp,adp2,bdp2,tau_camk,tau_k,w,b,I,th,sig_n, max_con,rho_ini,1.0,NE, Ip, con_M, con_M_in, con_ex_ex, con_ex_in, con_in_ex, con_in_in, Tw,Tp, ex,ex_in, seed, init,con_log, cv_th)
                                                            
                                                            
                                                            crw_path = cr_drc +"taupre{}_taupost{}_smaxcon{}_smaxampa{}_c_r_w.txt".format(tau_ca_pre,tau_ca_post,max_con, 1.0)
                                                            crw_path2 = cr_drc +"taupre{}_taupost{}_smaxcon{}_smaxampa{}_c_r_w.txt".format(pars_lr_ori["tau_ca_pre"],pars_lr_ori["tau_ca_post"],max_con, 1.0)
                                                            if not os.path.exists(crw_path) and not os.path.exists(crw_path2):
                                                                print("{} is not found".format(crw_path))
                                                                print("{} is not found".format(crw_path2))
                                                                print("calc c_r_w")
                                                                subprocess.run([fline_cr], shell=True)
                                                                subprocess.run([py_cr], shell=True)
                                                    
                                                            try:
                                                                if os.path.exists(crw_path):
                                                                    c_r_f = open(crw_path, "r")
                                                                elif os.path.exists(crw_path2):
                                                                    c_r_f = open(crw_path2, "r")
                                                                lines = c_r_f.read()
                                                                ls=lines.split("\n")
                                                                cpre_m = ls[0]
                                                                cpost_m = ls[1]
                                                                cpre_r = float(ls[2])
                                                                
                                                                cpost_r  = round(float(ls[3]),5)
                                                


                                                                print("cpre_m", cpre_m)
                                                                print("cpre_r", cpre_r)
                                                                print("cpost_r", cpost_r)

                                                                div = 0.7/float(cpre_m)
                                                            
                                                            except:
                                                                flag=1
                                                                traceback.print_exc()
                                                                

                                                        # for th_p in th_ps:
                                                        if th_complement ==True:
                                                            th_p = pars_lr["th_p"]/div
                                                            pars_lr["th_p"] = round(th_p,4)
                                                            th_d = pars_lr["th_d"]/div
                                                            pars_lr["th_d"] = round(th_d,4)
                                                            # count +=1
                                                        else:
                                                            th_p = pars_lr["th_p"]
                                                            pars_lr["th_p"] = round(th_p,4)
                                                            th_d = pars_lr["th_d"]
                                                            pars_lr["th_d"] = round(th_d,4)
                                                         
                                                        
                                                        print("th_p", th_p)
                                                        print("th_d", th_d)
                                                        for tau_s in tau_ss:
                                                            if tau_s != 0.0:
                                                                taus_inv=1/tau_s
                                                                pars_lr["tau_s"] =f"{taus_inv:,.8f}" 
                                                                # tau_s =  pars_lr["tau_s"] 
                                                                
                                                            else:
                                                                pars_lr["tau_s"] = 0.0
                                                        
                                                           
                                                            params_lr_str = str(cpre_r) + " " + str(cpost_r) +" " 
                                                            for k, v in pars_lr.items():
                                                                params_lr_str += str(pars_lr[k])
                                                                params_lr_str+= " "

                                        
                                                            fline = out +" "+ params_str+str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+" " +params_lr_str +" "+str(r_ini)+" "+str(a_ini)+" " +str(z_ini)+" " +str(alpha)+" "+str(beta)+" "+str(adp)+" "+str(bdp)+" " +str(adp2)+" "+str(bdp2)+" "+ str(tau_camk) +" "+ str(tau_k) +" " +str(w)+ " "+str(b)+" "+str(I)+" "+str(th)+ " "+str(sig_n)+" "+str(max_con) +" "+str(rho_ini)+" " +str(smin_ampa) +" " +str(smax_ampa) +" " +str(T) +" " +str(Tp) +" "+str(ex)+" "+str(ex_in)+" "+str(seed) + " " +init +" "+  date_i+  " "+ model_name + " " +str(blockdim_x)
                                                            print(fline) 
                                
                                                            if do_cargs==True:
                                                                f0.write("{} \n ".format(fline))
                                                                subprocess.run([fline], shell=True)

                                                            
                                                            
                                                            print("analyze lr")
                                                            if do_cargs==True:
                                                                # if not os.path.exists(npyf):
                                                                py_com = "python3 analysis_net_dynamics.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} &".format(model_name, date_i,cpre_r,cpost_r,pars_lr["th_p"],pars_lr["th_d"],pars_lr["gp"],pars_lr["gd"],pars_lr["tau_ca_pre"],pars_lr["tau_ca_post"],pars_lr["sig"],pars_lr["tau_s"],  r_ini, a_ini, z_ini, alpha, beta, adp,bdp,adp2,bdp2,tau_camk,tau_k,w,b,I,th,sig_n, max_con,rho_ini,smin_ampa,smax_ampa,NE, Ip, con_M, con_M_in, con_ex_ex, con_ex_in, con_in_ex, con_in_in, T,Tp, ex,ex_in, seed, init,con_log, cv_th,rho_sw_com, LA, UA, ti, td)
                                                                f0.write(py_com+'\n')
                                                                print(py_com)
                                                                subprocess.run([py_com], shell=True)
                                                                subprocess.run(['echo start analyze!\n'], shell=True)
                                                            else:
                                                                py_com = "python3 analysis_net_dynamics.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(model_name, date_i,cpre_r,cpost_r,pars_lr["th_p"],pars_lr["th_d"],pars_lr["gp"],pars_lr["gd"],pars_lr["tau_ca_pre"],pars_lr["tau_ca_post"],pars_lr["sig"],pars_lr["tau_s"],  r_ini, a_ini, z_ini, alpha, beta, adp,bdp,adp2,bdp2,tau_camk,tau_k,w,b,I,th,sig_n, max_con,rho_ini,smin_ampa,smax_ampa,NE, Ip, con_M, con_M_in, con_ex_ex, con_ex_in, con_in_ex, con_in_in, T,Tp, ex,ex_in, seed, init,con_log, cv_th,rho_sw_com, LA, UA, ti, td)
                                                                f0.write(py_com+'\n')
                                                                print(py_com)
                                                                subprocess.run([py_com], shell=True)
                                                                
                                                                
                                                              
                                                                
                                                                   
f0.close()

print("sh {}".format(sh_f))

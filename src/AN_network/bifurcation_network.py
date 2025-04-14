
import os 
import sys
import numpy as np
import os
import sys
import subprocess 
import pandas as pd

#automatically start to conduct bifurcation analysis in network models using the results of bifurcation analysis in single neuron

#python3 bifurcation_network.py nav kva kvsi kir nmdar gabar (1st or 2nd bifurcation analysis) bifurcated_molecule

fs = 25
fs_l = 20
#python3 bifurcation_network.py 1 1 1 1 1 1 1st cav 
#python3 bifurcation_network.py 1 1 1 1 1 1 1st pre
#python3 bifurcation_network.py 1 1 1 1 1 1 1st nmdar

except_dates =["2024_3_27_0","2024_5_14_3", "2024_3_9_0", "2024_4_20_0", "2024_3_27_1", "2024_4_26_0", "2024_5_18_0", "2024_4_6_0"]# 
nowdir = os.getcwd()+"/"
savedir = nowdir + "cc_dir/"
os.makedirs(savedir, exist_ok=True)

  #command for GPU analysis
op = op_param = "_net_bifurcation"
th_num= 2#default thread num
auto_th_num =False

param_cond_dir ="./net_col/"
#connection params
NE = 64
Ip = 20

con_M =1.0
con_M_in=1.0
con_ex_ex = 0.01  #ex to ex  #conSD
con_ex_in = 0.01  #ex to in 
con_in_in = 0.01    #conSDin
con_in_ex = 0.01

init = "r"
seed = 4
con_log = True

#--------------------------------------------------------

args = sys.argv

b_num= args[7]
bifur = "_"+(args[8])
avecon=1
nav = int(args[1])
kva = int(args[2])
kvsi = int(args[3])
kir = int(args[4])
nmdar = int(args[5])
gabar = int(args[6])

if bifur == "_cav" or  bifur == "_nmdar" or bifur == "_pre":
        g_str = "g"+bifur 


if b_num=="1st":
    T = 2500 #ms
    Tp= 500
    if bifur=="_nmdar" or bifur=="_cav":
        exp_ini = -2.0
        exp_last = 2.25
        exp_step = 0.25
        dt = 0.02

    elif bifur=="_pre":
        exp_ini = -2.0
        exp_last = 1.2
        exp_step = 0.2
        dt = 0.04

elif b_num=="2nd":
    T = 5000 #ms
    Tp= 1000
    
    if bifur=="_nmdar" or bifur=="_cav":
        exp_ini = -2.0
        exp_last = 1.6
        exp_step = 0.1
        dt = 0.02
    elif bifur=="_pre":
        exp_ini = -2.0
        exp_last = 0.9
        exp_step = 0.1
        dt = 0.04

exs = np.arange(exp_ini, exp_last, exp_step)

if auto_th_num == True:
    th_num =len(exs)


if nav==1 and kva ==1 and kvsi==1 and kir==1 and gabar==1: #NaK1
    keys = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 'g_cav', 'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_nmdar', 'g_gabar', 't_ca']
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
elif nav==0 and kva ==0 and kvsi==0 and kir==0 and gabar==1: #K2
    keys = ['g_leak', 'g_kvhh',  'g_cav', 'g_kca', 'g_nap', 'g_ampar','g_nmdar', 'g_gabar', 't_ca']
    ext = "K0"    


model_name_pre =  ext  # random search results data 
print(model_name_pre)
model_fols = os.listdir(nowdir+"results_bifurcation/")

model_pre=ext

model_name =model_name_pre + op+  bifur  #AN_net_bifurcation_nmdar
out = "./" + model_name_pre  + op + bifur

sh_f = nowdir + "net_search"+ "_"+model_name_pre + bifur+"_whole.sh"
fsh_all =  sh_f


fsh = open(fsh_all, "w")
prefix = "#!/usr/bin/bash"
fsh.write(prefix+'\n')

tbs=int(T/Tp)
NI = int(NE*Ip/(100-Ip))
c=0
if b_num == "1st":
    print("1st bifurcation analysis")
    for model_fol in model_fols:
        if len(model_fol.split("_")) <5:
            continue
        date_i = "_".join(model_fol.split("_")[0:4])
        model_name_pre2 = model_fol.split("_")[4]  #AN
        # print(model_name_pre2 )
        if model_name_pre == model_name_pre2:
        
            bifur_fols = os.listdir(nowdir+"results_bifurcation/"+model_fol+"/")
            pre_bifurs =[]
            for k, bifur_fol in enumerate(bifur_fols):
                if k>0:
                    continue
                p = (bifur_fol.split("_"))
                date="_".join(p[0:4])
                if date in except_dates:
                    print("remove {}".format(date))
                    continue

                print(p )
                if len(p)>=6 and p[5] =="g":
                    
                    g_str_bifur = p[5] + "_"+p[6]
                    pre_bifurs.append(g_str_bifur)

            if g_str in pre_bifurs:
                print(model_fol+"_"+g_str)
                f_bifurcated = "bifurcation_checked.xlsx"
                print(nowdir+"results_bifurcation/"+model_fol+"/"+model_fol+"_"+g_str+"/" + f_bifurcated)
                if not os.path.exists(nowdir+"results_bifurcation/"+model_fol+"/"+model_fol+"_"+g_str+"/" + f_bifurcated):
                    print("no checked fies:", nowdir+"results_bifurcation/"+model_fol+"/"+model_fol+"_"+g_str+"/" + f_bifurcated)
                    continue
                    
                df_params = pd.read_excel(nowdir+"results_bifurcation/"+model_fol+"/"+model_fol+"_"+g_str+"/" + f_bifurcated, engine="openpyxl")
            
                
                if len(df_params)==0:
                    print("no param sets")
                else:
                    
                   
                    for i in range(len(df_params)):
                        if i>0:
                            continue
                        
                          
                        date_i=date+"_"+str(i)
                        
                    
                        df_param = df_params.iloc[i,1:]
                        pars= df_param.to_dict()
                
                        order = i
                        params_str = ""
                        for k in keys:
                            params_str += str(pars[k])
                            params_str+= " "
                        pars_in = pars.copy()

                        
                        params_str = ""
                        for k in keys:
                            params_str += str(pars[k])
                            params_str+= " "
                        # params_in_str = ""
                        # for k in keys:
                        #     params_in_str += str(pars_in[k])
                        #     params_in_str+= " "
                    
                    #cc_dir_check()
                        
                        cc= savedir+"/{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}".format(model_name, date_i, NE,NI,con_M,con_ex_ex,con_M_in, con_in_in,T)
                            
                        if  not os.path.exists(cc + "_cc.xlsx") :
                            # print(True)s
                            
                            print("bifur_net: {}_{} will be analyzed".format(model_fol+"/"+g_str, i))

                            flag=0
                            # if gabaup == True:
                            fline = out +" " + params_str+str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+ " "+str(T) +" " +str(Tp) +" "+str(exp_ini)+" "+str(exp_last)+" "+str(exp_step)+" "+str(th_num)+" "+str(seed) + " " +init +" "+  date_i+ " "+ model_name + " " +str(NE+NI)
 
                            fsh.write(fline+'\n')
                            print("fline: ", fline)
                            py_com = "python3 analysis_network_wave.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} &".format(model_name, date_i, NE, Ip, con_M, con_M_in, con_ex_ex, con_ex_in, con_in_ex, con_in_in, T, Tp, exp_ini, exp_last, exp_step,  seed, init, con_log)
                            fsh.write('echo start analyze!\n')
                            fsh.write(py_com+'\n')
                            print("py_com: ", py_com)
                        else:
                            flag=1
                            print("bifur_net: {}_{} already analyzed".format(model_fol+"/"+g_str, i))
elif b_num=="2nd":
    postfix = "NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}".format(NE,NI,con_M,con_ex_ex,con_M_in,con_in_in,2500)
                
    model_param = model_pre  +op_param + bifur
    df_params = pd.read_excel(param_cond_dir+model_param+"_"+postfix+"_bifur_params.xlsx", engine="openpyxl")
    if len(df_params)==0:
        print("no param sets")
    else:
        
       
        for i in range(len(df_params)):
            df_param = df_params.iloc[i,1:14]
            date_i=df_params.iloc[i]["order"]
            date = "_".join(date_i.split("_")[0:4])
            
            print(date_i)
            if date in except_dates:
                print("remove {}".format(date))
                continue
        
            df_param = df_params.iloc[i,1:]
            pars= df_param.to_dict()
    
            order = i
            params_str = ""
            for k in keys:
                params_str += str(pars[k])
                params_str+= " "
            pars_in = pars.copy()
        
        
            
            
            params_str = ""
            for k in keys:
                params_str += str(pars[k])
                params_str+= " "
            # params_in_str = ""
            # for k in keys:
            #     params_in_str += str(pars_in[k])
            #     params_in_str+= " "
        
        #cc_dir_check()
            
            cc= savedir+"/{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}".format(model_name, date_i, NE,NI,con_M,con_ex_ex,con_M_in, con_in_in,T)
                
            if  not os.path.exists(cc + "_cc.xlsx") :
                # print(True)s
                
        

                flag=0
                # if gabaup == True:
                fline = out +" "+ params_str +str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+ " "+str(T) +" " +str(Tp) +" "+str(exp_ini)+" "+str(exp_last)+" "+str(exp_step)+" "+str(th_num)+" "+str(seed) + " " +init +" "+  date_i+ " "+ model_name + " " +str(NE+NI)

                fsh.write(fline+'\n')
                print("fline", fline)
                py_com = "python3 analysis_network_wave.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} &".format(model_name, date_i, NE, Ip, con_M, con_M_in, con_ex_ex, con_ex_in, con_in_ex, con_in_in, T, Tp, exp_ini, exp_last, exp_step,  seed, init, con_log)
                fsh.write('echo start analyze!\n')
                fsh.write(py_com+'\n')
            else:
                flag=1
                                            # print("bifur_net: {}_{} already analyzed".format(model_fol+"/"+g_str, i))




fsh.write("wait\n")
fsh.close()
                                                    
outc= "sh {}".format(sh_f)
print("doing sh {}".format(sh_f))
subprocess.run([outc], shell=True)

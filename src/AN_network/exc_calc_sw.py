
import os
import subprocess
import sys
from multiprocessing import Pool

import pandas as pd

mode="w"

args = sys.argv

op = "_net_effi"
op_cr = "_net_crw"
model_op=op
op_param = "_net_bifurcation"


re_dir = "./con_cu_i"
c_r_auto=True

cpre_r = 1.0
cpost_r =1.0

rho_ini = 0.5
smin_ampa = 0.5
smax_ampa=1.5

T = 60000  #ms
Tp = 10000  #ms

dt = 0.05
Tw = 10000  #ms
Tpw = 5000  #ms
offset_w =10000
dtw=0.01
offset =0

nav=1
kva =1
kvsi=1
kir=1
nmdar=1
gabar=1

cores=2

def multi_sw(cores, sh_fs):
    args = []
    for i, sh_f in enumerate(sh_fs):
        args.append((i, sh_f))
    print(f'analyze networks: using {cores} cores')
    with Pool(processes=cores) as pool:
        pool.map(sw_gpu, args)
        
def sw_gpu(args):
    i, sh_f = args
    print("doing: {}".format(sh_f))
    subprocess.run(["sh {}".format(sh_f)], shell=True)

args = sys.argv
c=1

lr = args[c]
c+=1
lr_p = args[c]
c+=1

comp = 0.0

model_name = args[c]
c+=1
date_i = args[c]
c+=1
print("date_i", date_i)

NE = int(args[c])
c+=1
Ip= float(args[c])
c+=1

con_M = float(args[c])  #[0.5, 1.0, 1.2, 1.5,1.7]   1.5, 2.1, 3.0 (N increase)
c+=1
con_M_in =  float(args[c])
c+=1
print("sw_ana")

con_ex_ex = float(args[c])
c+=1
con_ex_in = float(args[c])
c+=1
con_in_in = float(args[c])
c+=1
con_in_ex =float(args[c])
c+=1

T2=int(args[c])
c+=1
Tp=int(args[c])
c+=1
# dt = 0.02
con_log = True


seed = int(args[c])
c+=1

init =(args[c])
c+=1



ex_w = float(args[c])
c+=1
ex_s = float(args[c])
c+=1

bifur = args[c]
c+=1

# print("search_bifur", bifur)
# print("ex_w", ex_w)

NI = int(NE*Ip/(100-Ip))

ex_diff=0


ex_in_w = ex_w #-2.0
ex_in_s = ex_s

blockdim_x=NE+NI
postfix = "NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}".format(NE,NI,con_M,con_ex_ex,con_M_in,con_in_in,2500)


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
    
#copy to next
model_pre = ext 

param_cond_dir ="./net_col/"
model_param = model_pre  +op_param + bifur
df_params = pd.read_excel(param_cond_dir+model_param+"_"+postfix+"_bifur_params.xlsx", engine="openpyxl")
df_params_i=df_params[df_params["order"]==date_i]

df_param = df_params_i.iloc[0,1:14]

pars= df_param.to_dict()
print("pars",  pars)


lr_dir ="./lr_params_con/{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}/".format(model_name, date_i, NE,NI,con_M,con_ex_ex,con_M_in,con_in_in)
    
        
net_lr_ok =False

exs_lr =  os.listdir(lr_dir)

for ex_lr in exs_lr:
    if "xlsx" in ex_lr:
        continue
    lr_dir2 = lr_dir +str(ex_lr) + "/"
    if os.path.exists(lr_dir2+lr+"_"+lr_p +".xlsx"):
        net_lr_ok =True
        break
        

if net_lr_ok == True:
   

    lr_params = pd.read_excel(lr_dir+str(ex_lr)+"/"+lr+"_"+lr_p +".xlsx")

    model_name_lr = model_pre + model_op + bifur + "_"+lr + "_w"
    model_name_lr_s = model_pre+ model_op + bifur + "_"+lr + "_s"
    model_cr = model_pre  + op_cr + bifur
    out = "./"+model_pre + op + bifur

    nowdir = os.getcwd()+"/"

    res_dir = "./con_cu_i/{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/sminampa{}_smaxampa{}/{}_{}/{}/sw_end.csv".format(model_name_lr, NE, NI,con_M, con_ex_ex,con_M_in, con_in_in,smin_ampa,smax_ampa,lr,lr_p,date_i)
    if not os.path.exists(res_dir):

        sh_f_w = date_i+"_"+ lr +"_"+lr_p + op+bifur +"_ex"+str(ex_w)+".sh"
        sh_f_s = date_i+"_"+ lr +"_"+lr_p + op+bifur  +"_ex"+str(ex_s)+".sh"


        mode = "w" # "a" or "w"
        ftxt_w =nowdir + sh_f_w
        ftxt_s = nowdir + sh_f_s

        #crw save
        if mode == "w":
            f0 = open(ftxt_w, "w")
            pre = "#!/usr/bin/bash"
            f0.write(pre+'\n')

            fs = open(ftxt_s, "w")
            pre = "#!/usr/bin/bash"
            fs.write(pre+'\n')
        else:
            f0 = open(ftxt_w, "a")
            fs = open(ftxt_s, "a")

        params_str = ""
        for k in keys:
            params_str += str(pars[k])
            params_str+= " "

        ex_lr = ex_lr[2:]
        for i in range(len(lr_params)):
            if i> 0:
                continue
            pd_i = lr_params.iloc[i,:]
            pars_lr = pd_i.to_dict()




            gp = pars_lr["gp"]
            gd = pars_lr["gd"]
            tau_ca_pre = pars_lr["tau_ca_pre"]
            tau_ca_post = pars_lr["tau_ca_post"]
            sig = pars_lr["sig"]
            tau_s = pars_lr["tau_s"]
        #     order= offset_lr + i#par_i["order"]


            gp_s = str(gp)
            gd_s = str(gd)
            tau_pre_s = str(tau_ca_pre)
            tau_post_s = str(tau_ca_post)
            sig_s = str(sig)
            tau_s_s = str(tau_s)
            order = str(int(i))
            if c_r_auto==True:
                if con_log==False:
                    cr_drc ="./con_cu_i/{}/param_{}/cprer{}_cpostr{}_taupre{}_taupost{}/NE{}_NI{}/con_th{}_{}_{}_{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/cr_ex{}_exin{}/".format(model_cr, date_i, 1.0, 1.0, tau_ca_pre,tau_ca_post,NE, NI, con_ex_ex,con_ex_in,con_in_ex,con_in_in, seed,init,Tw,Tpw,dtw,ex_lr,ex_lr)
                elif con_log==True:
                    cr_drc ="./con_cu_i/{}/param_{}/cprer{}_cpostr{}_taupre{}_taupost{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/cr_ex{}_exin{}/".format(model_cr, date_i,  1.0, 1.0,  tau_ca_pre,tau_ca_post,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in, seed,init,Tw,Tpw,dtw,ex_lr,ex_lr)
                    print("cr_drc", cr_drc)
                out_c_r = "./"+model_pre+op_cr+bifur+" "+params_str+ str(1.0)+" " + str(1.0)+" "+str(tau_ca_pre)+" "+str(tau_ca_post)+" "+ str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+ " "+str(Tw) +" " +str(Tpw) +" "+str(Tw)+ " "+ str(dtw)+" "+str(ex_lr)+" "+str(ex_lr)+" "+str(seed) + " " +init +" "+  date_i+  " "+ model_cr + " " +str(blockdim_x)
                

                if not os.path.exists(cr_drc+"c_r_w.txt"):
                    print("do {}".format(out_c_r))
                    subprocess.run([out_c_r], shell=True)

                else:
                    print("{} is already done".format(i))
                c_r_f = open(cr_drc+"c_r_w.txt", "r")
                lines = c_r_f.read()
                ls=lines.split("\n")
                cpre_m = ls[0]
                cpost_m = ls[1]
                cpre_r = float(ls[2])
                cpost_r  = round(float(ls[3]),4)
                print("cpre_m", cpre_m)
                print("cpre_r", cpre_r)
                print("cpost_r", cpost_r)


                pre_fix = "./"+model_pre + op +bifur+" "+ params_str+ str(cpre_r)+" " + str(cpost_r)+" "
                
                if bifur=="_cav":
                    post_fix_w = str(rho_ini)+" "+str(smin_ampa)+" "+str(smax_ampa)+" "+ str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+ " "+str(T) +" " +str(Tp) +" "+str(ex_lr)+" "+str(ex_w)+" "+str(ex_in_w)+" "+str(seed) + " " +init +" "+  date_i+  " "+ model_name_lr + " " +str(NE+NI)
                    post_fix_s = str(rho_ini)+" "+str(smin_ampa)+" "+str(smax_ampa)+" "+ str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+ " "+str(T) +" " +str(Tp) +" "+str(ex_lr)+" "+str(ex_s)+" "+str(ex_in_s)+" "+str(seed) + " " +init +" "+  date_i+  " "+ model_name_lr_s + " " +str(NE+NI)
                else:
                    post_fix_w = str(rho_ini)+" "+str(smin_ampa)+" "+str(smax_ampa)+" "+ str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+ " "+str(T) +" " +str(Tp) +" "+str(ex_w)+" "+str(ex_in_w)+" "+str(seed) + " " +init +" "+  date_i+  " "+ model_name_lr + " " +str(NE+NI)
                    post_fix_s = str(rho_ini)+" "+str(smin_ampa)+" "+str(smax_ampa)+" "+ str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+ " "+str(T) +" " +str(Tp) +" "+str(ex_s)+" "+str(ex_in_s)+" "+str(seed) + " " +init +" "+  date_i+  " "+ model_name_lr_s + " " +str(NE+NI)

                # cpre_m = 0.083
                div = 0.7/float(cpre_m)

                th_p = pars_lr["th_p"]/div
                th_d = pars_lr["th_d"]/div
                th_p_s = str(th_p)

                th_d_s = str(th_d)


                lr_f = th_p_s + " " + th_d_s + " " + gp_s + " " + gd_s + " " + tau_pre_s + " " + tau_post_s + " " + sig_s + " " + tau_s_s + " " + lr + " "+ lr_p +" "
                print(pre_fix + lr_f + post_fix_w)
                
                
                drc ="./con_cu_i/{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/sminampa{}_smaxampa{}/{}_{}/".format(model_name_lr,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in,smin_ampa,smax_ampa,lr,lr_p)
                drc_s ="./con_cu_i/{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/sminampa{}_smaxampa{}/{}_{}/".format(model_name_lr_s,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in,smin_ampa,smax_ampa,lr,lr_p)
                
                drc_i = drc + date_i +  "/fre_exs/{}.csv".format(ex_w)#f"{ex_w:,.2f}")
                drc_i2 = drc + date_i +  "/fre_exs/{}.csv".format(f"{ex_w:,.2f}")
                drc_i_s = drc_s + date_i +"/fre_exs/{}.csv".format(ex_s)#(f"{ex_s:,.2f}")
                drc_i_s2 = drc_s + date_i +"/fre_exs/{}.csv".format(f"{ex_s:,.2f}")
                
                if not os.path.exists(drc_i) and not os.path.exists(drc_i2):
                    f0.write(pre_fix + lr_f + post_fix_w+'\n')
                else:
                    f0.write("echo {}_{}_already_exists".format(date_i, ex_w)+'\n')
                
                
                if not os.path.exists(drc_i_s) and not os.path.exists(drc_i_s2):
                    fs.write(pre_fix + lr_f + post_fix_s+'\n')
                else:
                    fs.write("echo {}_{}_already_exists".format(date_i, ex_s)+'\n')

        f0.write("wait\n")
        f0.close()

        fs.write("wait\n")
        fs.close()

        sh_fs=[sh_f_w, sh_f_s]

        if __name__ == "__main__":
            multi_sw(cores, sh_fs)

    else:
        print("{}: is already exists".format(res_dir))    
# subprocess.run(["chmod 777 -R /mnt/gpu_data/"], shell=True)


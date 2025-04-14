
import os
import sys
import numpy as np
sys.path.append('../')
sys.path.append('../anmodel')
import anmodel


v_off=0 #ms
T_w = "None" # None analyze in whole time
cv_w = 50 #ms
moving = 50 #ms 

cores = 10  #analyse using cores
cores_auto =False

calc = True
graph_show = False
pdf_save = False
stat = False
raster=False

ISI_M_r = [1.3, 3.0]
counts = 100 # * 10000 samples  ~100

nowdir = os.getcwd()+"/"
savedir = nowdir + "cc_dir/"
                               
                                               
if __name__=="__main__":
#
    args = sys.argv
    vc = 1

    # params for analysis
    model_name = args[vc]
    vc+=1
    date_i = (args[vc]) #[16, 32, 64, 128]
    vc+=1

    NE = int(args[vc]) #[16, 32, 64, 128]
    vc+=1
    Ip = float(args[vc])
    vc+=1
    con_M = float(args[vc])
    vc+=1
    con_M_in= float(args[vc])
    vc+=1
    con_ex_ex = float(args[vc])  #ex to ex  #conSD
    vc+=1
    print("con_ex_ex", con_ex_ex)
    con_ex_in = float(args[vc])  #ex to in 
    vc+=1
    con_in_ex = float(args[vc])    #conSDin
    vc+=1
    con_in_in = float(args[vc]) 
    vc+=1
    
    T =  int(args[vc])
    vc+=1
    
    Tp=  int(args[vc])
    vc+=1
   
    # print("dt", dt)
    exp_ini=float(args[vc]) 
    vc+=1 
    exp_last=float(args[vc]) 
    vc+=1
    exp_step = float(args[vc])
    vc+=1
   
    seed =  int(args[vc])
    vc+=1
    init = (args[vc])
    vc+=1
    con_log = (args[vc])
    vc+=1
  

    ex_diff = 0
    exs = np.arange(exp_ini, exp_last, exp_step).tolist()
    exs = [f"{ex:,.2f}" for ex in exs]
    for n, i in enumerate(exs):
        if i =="-0.00":
            exs[n] = "0.00"
            
    if cores_auto ==True:
        cores = len(exs)
   
    

    NI = int(NE*Ip/(100-Ip))
    tbs=int(T/Tp)
    print(con_log)
    if con_log=="False":
        drc ="./con_cu_i/{}/param_{}/NE{}_NI{}/con_th{}_{}_{}_{}/seed_{}/init_{}/T_{}/Tp_{}/".format(model_name,date_i, NE,NI, con_ex_ex,con_ex_in,con_in_ex,con_in_in, seed,init,  T, Tp, con_log)
       
    elif con_log=="True":
        drc ="./con_cu_i/{}/param_{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/seed_{}/init_{}/T_{}/Tp_{}/".format(model_name,date_i, NE, NI,con_M, con_ex_ex,con_M_in, con_in_in, seed, init, T, Tp)
       
        
    wave_check = anmodel.analysis_multi_adt.eval_wave(model_name, date_i,NE,NI,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_ex, con_in_in, T, Tp, str(con_log),drc, savedir, exs, ex_diff)
                                                       
    if calc==True:
        wave_check.multi_cal(cores, exs, cv_w, moving, v_off, ISI_M_r, counts, T_w, stat, raster)
        df_cc= wave_check.make_df()
 

import os
import sys

import traceback

sys.path.append('../')
sys.path.append('../anmodel')
import anmodel

v_off=0 #ms

off = 0
move_cv = 50
w_cv = 50

move =500 #ms
window = 5000 #ms
n_conv=10
cv_th=6.0 

cores = 10  #analyse using cores
cores_auto =False

calc =True
graph_show = False
pdf_save = True
stat = False
raster=False

                                               
if __name__=="__main__":
#
    args = sys.argv
    vc = 1

    # params for analysis
    model_name = args[vc]
    vc+=1
    date_i = (args[vc]) #[16, 32, 64, 128]
    vc+=1
    
    cpre_r = float(args[vc])
    vc+=1
    cpost_r=float(args[vc])
    vc+=1
    th_p = (args[vc])
    vc+=1
    th_d = (args[vc])
    vc+=1
    gp =(args[vc])
    vc+=1
    gd =(args[vc])
    vc+=1
    tau_ca_pre =(args[vc])
    vc+=1
    tau_ca_post=(args[vc])
    vc+=1
    sig = (args[vc])
    vc+=1
    tau_s = (args[vc])
    vc+=1
    
    Sp_ini=float(args[vc]) 
    vc+=1
 
    K_ini=float(args[vc]) 
    vc+=1
    z_ini=float(args[vc]) 
    vc+=1
    
    alpha= float(args[vc])
    vc+=1
    beta= float(args[vc])
    vc+=1
   
    adp= float(args[vc])
    vc+=1
    bdp= float(args[vc])
    vc+=1
    adp2= float(args[vc])
    vc+=1
    bdp2= float(args[vc])
    vc+=1
    print("alpha", alpha)
    tau_camk= int(args[vc]) #[16, 32, 64, 128]
    vc+=1
    tau_k= int(args[vc]) #[16, 32, 64, 128]
    vc+=1
    w= float(args[vc])
    vc+=1
    b= float(args[vc])
    vc+=1
    I= float(args[vc])
    vc+=1
    th= float(args[vc])
    vc+=1
    sig_n= float(args[vc])
    vc+=1
  
    max_con=float(args[vc]) 
    vc+=1
    rho_ini=float(args[vc]) 
    vc+=1
    
    smin_ampa=float(args[vc]) 
    vc+=1
   
    smax_ampa=float(args[vc]) 
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
    # print("con_ex_ex", con_ex_ex)
    con_ex_in = float(args[vc])  #ex to in 
    vc+=1
    con_in_ex = float(args[vc])    #conSDin
    vc+=1
    con_in_in = float(args[vc]) 
    vc+=1
    print("NE", NE)
    
    
    T =  int(args[vc])
    vc+=1
    Tp=  int(args[vc])
    vc+=1

    ex=float(args[vc]) 
    vc+=1 
   
    ex_in =  float(args[vc])
    vc+=1
    
    seed =  int(args[vc])
    vc+=1
    init = (args[vc])
    vc+=1
    
    con_log = (args[vc])
    vc+=1
    
    cv_th =  float(args[vc])
    vc+=1
    
    rho_sw_com= (args[vc])
    vc+=1
    
    LA=  float(args[vc])
    vc+=1
    
    UA=  float(args[vc])
    vc+=1
    
    ti= int(args[vc]) #[16, 32, 64, 128]
    vc+=1
    
    td= int(args[vc]) #[16, 32, 64, 128]
    vc+=1
    #
    
        
    NI = int(NE*Ip/(100-Ip))
 
   
    tbs=int(T/Tp)
    print(con_log)
    if con_log=="False":
        
        drc ="./con_cu_i/"
       
    elif con_log=="True":
        drc ="./con_cu_i/{}/param_{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/Spini{}_Kini{}_zini{}/exp{}_{}_{}_{}_{}_{}/taucamk{}_tauk{}/w{}_b{}_I{}_th{}_sign{}/ex{}_exin{}/maxcon{}_sminampa{}_smaxampa{}/thp{}_thd{}_gp{}_gd{}_taupre{}_taupost{}_sig{}/tausinv_{}/seed_{}/init_{}/T{}_Tp{}/".format(model_name, date_i,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in,  Sp_ini, K_ini, z_ini, alpha,beta, adp,bdp,adp2,bdp2, tau_camk, tau_k,w,b,I,th,sig_n,ex,ex,max_con,smin_ampa,smax_ampa,th_p,th_d,gp,gd,tau_ca_pre,tau_ca_post,sig,tau_s,seed,init,T,Tp)
    drc_neu = drc + "neu/"
    drc_syn = drc + "syn/"
    
    
    
    fig_dir = "./cadyn_dir_lrsw/{}/param_{}/NE{}_NI{}/conM{}_SD{}_conMin{}_inSD{}/Spini{}_Kini{}_zini{}/exp{}_{}_{}_{}_{}_{}/taucamk{}_tauk{}/w{}_b{}_I{}_th{}_sign{}/T_{}/ex{}_exin{}/thp{}_thd{}_gp{}_gd{}_taupre{}_taupost{}_sig{}/maxcon{}_sminampa{}_smaxampa{}".format(model_name, date_i, NE,NI,con_M,con_ex_ex,con_M_in, con_in_in,Sp_ini,K_ini, z_ini, alpha,beta, adp,bdp,adp2,bdp2, tau_camk, tau_k,w,b,I,th,sig_n,T,ex,ex,th_p,th_d,gp,gd,tau_ca_pre,tau_ca_post,sig,max_con,smin_ampa, smax_ampa)
    try:
        os.makedirs(fig_dir)
    except FileExistsError:
        print("already exist")
            # print("{} is already exist".format(fig_dir + fol))
            
    if con_log == "True":
        savef = fig_dir +"/tausinv_{}".format(tau_s)

    
    
    if "nmdar" in model_name or "pre" in model_name :
        print("nmdar")
        wave_check = anmodel.analysis_lr_dyn.eval_wave(model_name, date_i, NE, NI,T, Tp, drc, fig_dir, savef, ex, ex_in,move, window, LA, UA, ti, td)
    elif "_cav" in model_name:  
        wave_check = anmodel.analysis_lr_dyn_cav.eval_wave(model_name, date_i, NE, NI,T, Tp, drc, fig_dir, savef, ex, ex_in,move, window, LA, UA, ti, td)
                                            
    
    if calc==True:
        try:
            if not os.path.exists(savef + "_cv_ori.bin"):
                wave_check.multi_cal(cores, drc_neu, NE+NI, ex,ex_in, T, move, window, move_cv, w_cv)
            sw_cv= wave_check.cv_th_data(n_conv, cv_th)
            df_sw, sleep_durs, wake_durs = wave_check.sw_time(move,sw_cv,cv_th)
       
            wave_check.rho_sw(sleep_durs, wake_durs)
        except:
            traceback.print_exc()
        
     
    if pdf_save ==True or graph_show == True:
        wave_check.graph_saving(pdf_save, graph_show)
    if raster ==True:
        try:
            wave_check.raster_plot(drc_neu, NE+NI, ex,ex_in,T, off)
        except:
            traceback.print_exc()
    
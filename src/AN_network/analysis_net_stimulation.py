
import os
import sys
import traceback
import numpy as np
sys.path.append('../')
sys.path.append('../anmodel')
import anmodel

v_off=0 #ms
# w = 3  #ms
#cv
off = 0
move_cv = 50
w_cv = 50
# T = 100000 #ms
move =500 #ms
window = 5000 #ms
n_conv=10
cv_th=6.0 #N=20  p<0.01
# cv_th = 13.2  #N=80   p<0.01


cores = 10  #analyse using cores
cores_auto =False

calc =False 
graph_show = False
pdf_save = True
stat = False
raster=False

ISI_M_r = [0.6, 3.3]
counts = 100 # * 10000 samples  ~100


#command
#python3 ana_net_data.py model_name i pars pars_in NE Ip con_M con_M_in con_ex_ex con_ex_in con_in_ex con_in_in g_nap_SD T Tp dt exs ex ex_in seed init
                                           
                                               
if __name__=="__main__":
#
    args = sys.argv
    vc = 1

    # params for analysis
    lr= args[vc]
    vc+=1
    lr_p = args[vc]
    vc+=1
    
    model_name = args[vc]
    vc+=1
    date_i = (args[vc]) #[16, 32, 64, 128]
    vc+=1
    
   
    th_p = float(args[vc])
    vc+=1
    th_d = float(args[vc])
    vc+=1
    gp =float(args[vc])
    vc+=1
    gd =float(args[vc])
    vc+=1
    tau_ca_pre =float(args[vc])
    vc+=1
    tau_ca_post=float(args[vc])
    vc+=1
    sig = float(args[vc])
    vc+=1
    tau_s = float(args[vc])
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
    Gex = int(args[vc]) #[16, 32, 64, 128]
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
   
    dt =  float(args[vc]) 
    vc+=1
    print("dt", dt)
    
    stim_start=  int(args[vc])
    vc+=1
    stim_dur=  int(args[vc])
    vc+=1
   
    stim_hz=  int(args[vc])
    vc+=1
    delay=  float(args[vc])
    vc+=1
    
    peak=  float(args[vc])
    vc+=1
    dur=  float(args[vc])
    vc+=1
    fy=  int(args[vc])
    vc+=1
    ly=  int(args[vc])
    vc+=1
    vrest=  int(args[vc])
    vc+=1
    
    sws_start=  int(args[vc])
    vc+=1
    sws_dur=  int(args[vc])
    vc+=1
    
    # print("dt", dt)
    ex=float(args[vc]) 
    vc+=1 
    print("ex", ex)
   
    ex_in =  float(args[vc])
    vc+=1
    
    ex_s=float(args[vc]) 
    vc+=1 
    print("ex_s", ex_s)
    
    seed =  int(args[vc])
    vc+=1
    init = (args[vc])
    vc+=1
    
    con_log = (args[vc])
    vc+=1
    print("init", init)
    
    LA= 1.0
    UA=  0.0
    ti=10000
    td= 10000


    NI = int(NE*Ip/(100-Ip))

   
    time = np.arange(0, T , dt)/1000 #s

 
    par_file = "param_file.txt"
    tbs=int(T/Tp)
    print(con_log)
    if con_log=="False":
        
        drc ="./con_cu_i/{}/NE{}_NI{}_Gex{}/conpro{}_{}_{}_{}/sminampa{}_smaxampa{}/{}_{}/{}/exs/{}/stim{}_{}_{}hz_delay{}_{}_{}_{}_{}_{}/sws{}_{}_{}/seed_{}/init_{}/T_{}/Tp_{}/".format(model_name, NE, NI,Gex,con_ex_ex,con_ex_in, con_in_ex, con_in_in, smin_ampa, smax_ampa, lr, lr_p, date_i, ex, stim_start, stim_dur, stim_hz,delay,peak, dur,fy,ly,vrest,sws_start,sws_dur,ex_s, seed,init,T,Tp)
        fig_dir = "./stim_dir/{}/param_{}/NE{}_NI{}_Gex{}/conpro{}_{}_{}_{}/T_{}/ex{}_exin{}/stim{}_{}_{}hz_delay{}_{}_{}_{}_{}_{}/sws{}_{}_{}/".format(model_name, date_i, NE,NI,Gex,con_ex_ex,con_ex_in, con_in_ex, con_in_in,T,ex,ex, stim_start, stim_dur, stim_hz,delay,peak, dur,fy,ly,vrest,sws_start,sws_dur,ex_s)
        
        
    elif con_log=="True":
    
        drc ="./con_cu_i/{}/NE{}_NI{}_Gex{}/conM{}_conSD{}_conMin{}_conSDin{}/sminampa{}_smaxampa{}/{}_{}/{}/exs/{}/stim{}_{}_{}hz_delay{}_{}_{}_{}_{}_{}/sws{}_{}_{}/seed_{}/init_{}/T_{}/Tp_{}/".format(model_name, NE, NI,Gex,con_M, con_ex_ex,con_M_in, con_in_in, smin_ampa, smax_ampa, lr, lr_p, date_i, ex, stim_start, stim_dur,  stim_hz,delay,peak, dur,fy,ly,vrest,sws_start,sws_dur,ex_s, seed,init,T,Tp)
        fig_dir = "./stim_dir/{}/param_{}/NE{}_NI{}_Gex{}/conM{}_conSD{}_conMin{}_conSDin{}/T_{}/ex{}_exin{}/stim{}_{}_{}hz_delay{}_{}_{}_{}_{}_{}/sws{}_{}_{}/".format(model_name, date_i, NE,NI,Gex,con_M,con_ex_ex,con_M_in, con_in_in,T,ex,ex, stim_start, stim_dur, stim_hz,delay,peak, dur,fy,ly,vrest,sws_start,sws_dur,ex_s)
 
    drc_neu = drc + "neu/"
    drc_syn = drc + "syn/"
    
    
    
    try: 
        os.makedirs(fig_dir)
    except FileExistsError:
        print("already exist")
            # print("{} is already exist".format(fig_dir + fol))
            
    
    # savef = fig_dir +"thp{}_thd{}_gp{}_gd{}_taupre{}_taupost{}_sig{}_taus{}_sminampa{}_smaxampa{}".format(th_p,th_d,gp,gd,tau_ca_pre,tau_ca_post,sig, tau_s, smin_ampa, smax_ampa)
    savef = fig_dir +"sminampa{}_smaxampa{}".format(smin_ampa, smax_ampa)
       


 
        
    if "nmdar" in model_name or "pre" in model_name:
        wave_check = anmodel.analysis_lr_stim.eval_wave(model_name, date_i, NE, NI,T, Tp, dt,drc, fig_dir, savef, ex, ex_in,move, window, LA, UA, ti, td)
    elif "cav" in model_name:
        wave_check = anmodel.analysis_lr_stim_cav.eval_wave(model_name, date_i, NE, NI,T, Tp, dt,drc, fig_dir, savef, ex, ex_in,move, window, LA, UA, ti, td)
              
    
     
    if pdf_save ==True or graph_show == True:
        wave_check.graph_saving(pdf_save, graph_show)
        
    wave_check.weights_cal(Gex, stim_start, stim_dur)
    
    # wave_check.a_range()
    if raster ==True:
        try:
            wave_check.raster_plot(drc_neu, NE+NI, ex,ex_in,T, off)
        except:
            traceback.print_exc()
    

#make sh file
import os
import subprocess
import sys
from multiprocessing import Pool
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal import periodogram
from scipy import signal
import numpy as np
import pandas as pd

mode="w"


#python3 calc_net_stim.py cav

args=sys.argv

nowdir = os.getcwd() + "/"

op = "_net_stim"
op_cr = "_net_crw"
model_op=op 
op_param = "_net_bifurcation"
op_effi = "_net_effi"
bifur = "_"+args[1]

lr = "STDP"
lr_p ="a0.7t50th0.6"
model_file = nowdir+"net_lr_data.xlsx"  #item lists for stimulation

c_r_auto=True
do_cargs=False
Tw = 10000  #ms  time of preliminary simulation for scaling parameters
Tpw = 5000  #ms  time block of preliminary simulation for scaling parameters
offset_w =10000
dtw=0.01  #ms   step of preliminary simulation for scaling parameters
offset =0

lag=0.0
nav=1
kva =1
kvsi=1
kir=1
nmdar=1
gabar=1
cores=1



#connection parameters
NE = 64  # number of excitatory neurons
Ip = 20  # ratio of inhibitory neurons 
Gex = 20  # number of neurons in a group
NI = int(NE*Ip/(100-Ip)) #number of inhibitory neurons
con_log = True  #True -> lognormal connections, False -> random connections
con_M = 1.0  # mean for the lognormal distribution of the number of synapses per excitatory neuron
con_M_in =  1.0 # mean for the lognormal distribution of the number of synapses per inhibitory neuron
log=0.01
con_ex_ex = log # SD for the lognormal distribution of the number of synapses in Ex-Ex connections
con_ex_in = log  # SD for the lognormal distribution of the number of synapses in Ex-In connections
con_in_in = log  # SD for the lognormal distribution of the number of synapses in In-In connections
con_in_ex = log  # SD for the lognormal distribution of the number of synapses in In-Ex connections


T = 45000  # ms  #total simulation time
Tp = 500

T2 = 5000
Tp2=1000

stim_start=15000   #ms   time for start of stimulation

stim_durs= [15000]  #ms  duration of stimulation protocol

stim_hzs=[10, 20, 5]  #ms     Hz of stimulation

delays=[0.0] #ms      stimulatiom delay bwtween pre and post-synaptic neurons

#parameters for stimulation waveform
durs =[0.8] 
fys =[-68]#, -70, -65, -60]
lys=[-76]#, -80, -75]
vrests=[-70]
prs = [1.1, 1.2, 1.3, 1.4, 1.5]   #coefficients of peak voltage when stimulation
offset = 200
dt = 0.05
off = 2

sws_start=0   #ms delay time after stimulation
sws_dur=15000  #ms duration of SWS

exp_ex = 0.5
exp_in = 0.5
init = "r"
seed = 4
con_log = True

NI = int(NE*Ip/(100-Ip))

ex_diff=0

rho_ini = 0.5
smin_ampa = 0.5
smax_ampa=1.5
blockdim_x=NE+NI

args = sys.argv
c=1

def fre_spike_i(v, T, dt):
    
    if np.any(np.isinf(v)) or np.any(np.isnan(v)):
        print("fre_spike: nan")
        return "Excluded",0, 0, np.zeros(int(T/dt), dtype="int8")
    else:

        detv: np.ndarray = signal.detrend(v)
        max_potential: float = max(detv)
        f: np.ndarray  # Array of sample frequencies
        spw: np.ndarray  # Array of power spectral density or power spectrum
        f, spw = periodogram(detv, fs=1/dt*1000)
        maxamp: float = max(spw)
        nummax: int = spw.tolist().index(maxamp)
        maxfre: float = f[nummax]

        peaks = find_peaks(v, height=-20)[0]
#         print(peaks)
        peak_np = np.zeros(int(T/dt), dtype="int8")
        peak_np[peaks]=1
        return "not calc", maxfre, len(peaks), peaks
    
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
    
def gen_spikes(T, off, dt, n, interval,lag):

        spikes= np.zeros((2,int((T)/dt)))
        
        for i in range(n):
          
            if int(off/dt)+i*int(interval/dt) < int((T+off)/dt):
                spikes[0][int(off/dt)+i*int(interval/dt)] = 1
            if int(off/dt)+i*int(interval/dt) + int(lag/dt) < int((T+off)/dt):
                spikes[1][int(off/dt)+i*int(interval/dt) + int(lag/dt)]=1
       
        return spikes
    
def sp_to_mp_2(spikes, T, off ,peak, fx,fy, lx,ly, vrest):
    Lt = int((T)/dt)
    pre_x = np.arange(0,Lt,1)* spikes[0]
    pre_y = spikes[0]*peak
    post_x = np.arange(0,Lt,1)* spikes[1]
    post_y = spikes[1]*peak
    for m,j in enumerate(spikes[0]):
        if j==1:
            #if m+10 < spikes.shape[1]-2*off/dt:
                pre_x[m+lx]=pre_x[m] + lx
                pre_y[m+lx]= ly
            #if m-36 >=off/dt:
                pre_x[m+fx]=pre_x[m] + fx
                pre_y[m+fx]= fy

    for m,j in enumerate(spikes[1]):
        if j==1:
            #if m+10 < spikes.shape[1]-2*off/dt:#10000
                post_x[m+lx]=post_x[m] + lx
                post_y[m+lx]= ly
            #if m-36 >=off/dt:
                post_x[m+fx]=post_x[m] +fx
                post_y[m+fx]= fy
                

    pre_x_a = np.take(pre_x, np.nonzero(pre_x)[0])
    pre_y_a = np.take(pre_y, np.nonzero(pre_y)[0])
    post_x_a = np.take(post_x, np.nonzero(post_x)[0])
    post_y_a = np.take(post_y, np.nonzero(post_y)[0])

    af = (peak-fy)/(0-fx)
    xf_pre = int(pre_x_a[0]-(pre_y_a[0]-vrest)/af)
    xl_pre = int(pre_x_a[-1]+int(3/dt))  #3ms で過分極から戻る
    xf_post = int(post_x_a[0]-(post_y_a[0]-vrest)/af)
    xl_post = int(post_x_a[-1]+int(3/dt))
    pre_x_a = np.insert(pre_x_a,0,xf_pre)
    pre_x_a = np.append(pre_x_a,xl_pre)
    pre_y_a = np.insert(pre_y_a,0,vrest)
    pre_y_a = np.append(pre_y_a,vrest)
    post_x_a = np.insert(post_x_a,0,xf_post)
    post_x_a = np.append(post_x_a,xl_post)
    post_y_a = np.insert(post_y_a,0,vrest)
    post_y_a = np.append(post_y_a,vrest)

    x_latent_post = np.arange(min(post_x_a), max(post_x_a)+1, 1).astype("int32")
    x_latent_pre = np.arange(min(pre_x_a), max(pre_x_a)+1, 1).astype("int32")
    
     #ip1 = ["最近傍点補間", lambda x, y: interpolate.interp1d(x, y, kind="nearest",  fill_value="extrapolate")]
    ip2 = ["線形補間", lambda x, y: interpolate.interp1d(x,y,fill_value="extrapolate")]
    for method_name, method in [ip2]:
       # print(method_name)
        fitted_curve_pre = method(pre_x_a, pre_y_a)
        fitted_curve_post = method(post_x_a, post_y_a)
    middle_pre = fitted_curve_pre(x_latent_pre)
    middle_post = fitted_curve_post(x_latent_post)

    first_pre = np.ones(x_latent_pre[0])*vrest
    last_pre = np.ones(Lt-1-x_latent_pre[-1])*vrest
    first_post = np.ones(x_latent_post[0])*vrest
    last_post = np.ones(Lt-1-x_latent_post[-1])*vrest
    mp_pre = np.append(first_pre,middle_pre).astype("float64")
    mp_pre = np.append(mp_pre,last_pre).astype("float64")
    mp_post = np.append(first_post,middle_post).astype("float64")
    mp_post = np.append(mp_post,last_post).astype("float64")
   
    return mp_pre, mp_post


model_df = pd.read_excel(model_file , engine="openpyxl")
model_indexes = model_df["date_i"]


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

postfix = "NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}".format(NE,NI,con_M,con_ex_ex,con_M_in,con_in_in,2500)
#read parameter files for channel and receptor conductances
param_cond_dir ="./net_col/"
model_param = model_pre  +op_param + bifur
df_params = pd.read_excel(param_cond_dir+model_param+"_"+postfix+"_bifur_params.xlsx", engine="openpyxl")


model_name = model_pre + model_op + bifur
model_cr = model_pre  + op_cr + bifur
out = "./"+model_pre + op + bifur

nowdir = os.getcwd()+"/"

# if not os.path.exists(savef):
sh_f_w = "stim_"+ lr +"_"+lr_p + op+bifur +"_ex"+".sh"

mode = "w" # "a" or "w"
ftxt_w =nowdir + sh_f_w
# ftxt_s = nowdir + sh_f_s

#crw save
if mode == "w":
    f0 = open(ftxt_w, "w")
    pre = "#!/usr/bin/bash"
    f0.write(pre+'\n')

else:
    f0 = open(ftxt_w, "a")
    # fs = open(ftxt_s, "a")




tbs=int(T2/Tp2)

if not os.path.exists("./vstims/"):
    os.makedirs("./vstims/")

c=0
for i, date_i in enumerate(model_indexes):
    # if i >30:
    #     continue
    
    
    df_i = model_df.iloc[i,:] 
    
    if str(df_i["bifur"])!= args[1]:
        continue

    # ex_w = float(df_i["ex_w"])  #conductance coefficient for wake-like pattern
    # ex_s = float(df_i["ex_s"])  #conductance coefficient for sleep-like pattern
    # ex_ss = [ex_s] 
    model_name_lr = model_pre + op_effi + bifur +"_"+lr +"_w"
    res_dir = "./con_cu_i/{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/sminampa{}_smaxampa{}/{}_{}/{}/".format(model_name_lr,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in,smin_ampa,smax_ampa,lr,lr_p,date_i.replace("param_", ""))
    
    drc ="./con_cu_i/{}/param_{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/seed_{}/init_{}/T_{}/Tp_{}/".format(model_param, date_i, NE, NI,con_M, con_ex_ex,con_M_in, con_in_in,  seed, init, T2, Tp2)
   
    if not os.path.exists(res_dir+"sw_end.csv"):
        continue
    exs= np.genfromtxt(res_dir+"sw_end.csv", delimiter=',', dtype=np.float64)
    ex_w=exs[0]
    ex_w_str=f"{ex_w:,.2f}"
    if ex_w_str=="-0.00":
        ex_w_str="0.00"
    ex_in_w=ex_w
    ex_in_w_str=ex_w_str
    ex_ss = [exs[1]]

   
    for tb in range(tbs):
        v_file = drc+ "ex"+ex_w_str+"_"+"ex_in"+ex_w_str+"_"+"N"+str(0)+"_"+str(tb)+".bin" 
        f2= open(v_file, "rb")
        rectype = np.dtype(np.float64)
        v_tb = np.fromfile(f2, dtype=rectype).astype("float64")
        if tb==0:
            v=v_tb
        else:
            v=np.append(v, v_tb)  #5s data v
        f2.close()
        
    dt = round(T2/len(v), 4)
    pattern, maxfre, sp_c, peak_index= fre_spike_i(v, T2, dt)
    offset_peak_index = peak_index[np.where(peak_index > int(Tp/dt))[0]]
    v_peaks=np.take(v, offset_peak_index)

    peak_ori=  np.mean(v_peaks)



    
    print("date_i", date_i)
    print("peak_ori", peak_ori)
   
    date_orders = df_params["order"].tolist()

#         print(date_orders)
    if not date_i in date_orders:
        continue
    df_bifur_i=df_params[df_params["order"]==date_i]
#         if date_i=="2023_3_29_0_0":
#             print("df_bifur_i", df_bifur_i)
    pars_df= df_bifur_i.iloc[0,1:14]
    pars = pars_df.to_dict()
    
    
    print("pars",  pars)
    params_str = ""
    for k in keys:
        params_str += str(pars[k])
        params_str+= " "
    i=0
    
    # lr_dir ="./lr_params_con/{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_gnap{}_gabac{}/ex{}/".format(model_name, date_i, NE,NI,con_M,con_ex_ex,con_M_in,con_in_in,g_nap_SD,gaba_c, ex_w)

    lr_dir ="./lr_params_con/{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}/".format(model_param, date_i, NE,NI,con_M,con_ex_ex,con_M_in,con_in_in) 
    net_lr_ok =False
    exs_lr =  os.listdir(lr_dir)
    # try:
    #     exs_lr.remove('{}_{}.xlsx').format(lr, lr_p)
    # except:
    #     traceback.print_exc()
    for ex_lr in exs_lr:
        if "xlsx" in ex_lr:
            continue
        lr_dir2 = lr_dir +str(ex_lr) + "/"
        if os.path.exists(lr_dir2+lr+"_"+lr_p +".xlsx"):
            net_lr_ok =True
            pars_lr_ori=pd.read_excel(lr_dir2+lr+"_"+lr_p +".xlsx", engine="openpyxl", index_col=0)
            break
        
    if net_lr_ok !=True:
        continue
    
    print("pars_lr_ori", pars_lr_ori)
    pars_lr=(pars_lr_ori.T).to_dict()
    pars_lr = pars_lr[0]
    print("pars_lr", pars_lr)
    
    gp = round(pars_lr['gp'],3)
    gd =round(pars_lr['gd'],3)
    sig = round(pars_lr["sig"],4)
    tau_ca_pre = round(pars_lr['tau_ca_pre'],3)
    tau_ca_post =round(pars_lr['tau_ca_post'],3)
                                                        
    th_p = round(pars_lr["th_p"], 4)
    th_d = round(pars_lr["th_d"], 4)
    sig = round(pars_lr["sig"], 3)
    tau_s = round(pars_lr["tau_s"], 4)
    
    gp_s = str(gp)
    gd_s = str(gd)
    tau_pre_s = str(tau_ca_pre)
    tau_post_s = str(tau_ca_post)
    sig_s = str(sig)
    tau_s_s = str(tau_s)
    order = str(int(i))
    

    # for ex_w in ex_ws:

    if c_r_auto==True:
        if con_log==False:
            cr_drc ="./con_cu_i/{}/param_{}/cprer{}_cpostr{}_taupre{}_taupost{}/NE{}_NI{}/con_th{}_{}_{}_{}/seed_{}/init_{}/T_{}/Tp_{}/cr_ex{}_exin{}/".format(model_cr, date_i, 1.0, 1.0, tau_ca_pre,tau_ca_post,NE, NI, con_ex_ex,con_ex_in,con_in_ex,con_in_in, seed,init,Tw,Tpw,dtw,ex_w,ex_w)
        elif con_log==True:
            cr_drc ="./con_cu_i/{}/param_{}/cprer{}_cpostr{}_taupre{}_taupost{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/cr_ex{}_exin{}/".format(model_cr, date_i,  1.0, 1.0,  tau_ca_pre,tau_ca_post,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in, seed,init,Tw,Tpw,dtw,ex_w,ex_w)
        print("cr_drc", cr_drc)
        out_c_r = "./"+model_pre+op_cr+bifur+" "+params_str+ str(1.0)+" " + str(1.0)+" "+str(tau_ca_pre)+" "+str(tau_ca_post)+" "+ str(NE)+" "+str(Ip)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+ " "+str(Tw) +" " +str(Tpw) +" "+str(Tw)+ " "+ str(dtw)+" "+str(ex_w)+" "+str(ex_w)+" "+str(seed) + " " +init +" "+  date_i+  " "+ model_cr + " " +str(blockdim_x)
     
        if not os.path.exists(cr_drc+"c_r_w.txt"):
            print("do {}".format(out_c_r))
            subprocess.run([out_c_r], shell=True)

        else:
            print("{} is already done".format(out_c_r))
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
        
        div = 0.7/float(cpre_m)

        th_p = round(pars_lr["th_p"]/div,4)
        th_d = round(pars_lr["th_d"]/div,4)
        th_p_s = str(th_p)

        th_d_s = str(th_d)

    
    for stim_dur in stim_durs:
        # for peak in peaks:
        for pr in prs:
            peak = round(peak_ori*pr, 2)
            for dur in durs:
                for fy in fys:
                    for ly in lys:
                        for vrest in vrests:
                            for stim_hz in stim_hzs:
                                for delay in delays:
                                    for ex_s in ex_ss:
                                        fx = int(-dur/dt) #down to up
                                        lx=  int(dur/dt)  #down

                                        interval =int(1000/stim_hz)# 100 #ms
                                        n = int((stim_dur+offset)/interval) 

                                        stim_T = n *interval  

                                        if not os.path.exists("vstims/vpre_stim_{}hz_{}_{}_{}_{}_{}_{}_{}_{}".format(stim_hz, dt,stim_dur,lag, peak, dur,fy,ly,vrest)):
                                        
                                            spikes_stim = gen_spikes(stim_T,off,dt,n,interval,delay)
                                            vpre, vpost =sp_to_mp_2(spikes_stim, stim_T, off, peak, fx, fy, lx, ly, vrest)
                                    

                                            np.save("vstims/vpre_stim_{}hz_{}_{}_{}_{}_{}_{}_{}_{}".format(stim_hz, dt,stim_dur,lag, peak, dur,fy,ly,vrest), vpre[int(offset/dt):int((stim_dur+offset)/dt)]) 
                                            np.save("vstims/vpost_stim_{}hz_{}_{}_{}_{}_{}_{}_{}_{}".format(stim_hz, dt,stim_dur,lag, peak, dur,fy,ly,vrest), vpost[int(offset/dt):int((stim_dur+offset)/dt)]) 



                                        pre_fix = out+" "+ params_str+ str(cpre_r)+" " + str(cpost_r)+" "
                                        post_fix_w = str(rho_ini)+" "+str(smin_ampa)+" "+str(smax_ampa)+" "+ str(NE)+" "+str(Ip)+" "+str(Gex)+" "+str(con_M)+" "+str(con_M_in)+" "+str(con_ex_ex)+" "+str(con_ex_in)+" "+str(con_in_ex)+" "+str(con_in_in)+ " "+str(T) +" " +str(Tp) + " "+ str(dt)+" "+ str(stim_start)+" " + str(stim_dur)+" "  + str(stim_hz)+" "+ str(delay)+" "+ str(peak)+" "+ str(dur)+" "+ str(fy)+" "+ str(ly)+" "+ str(vrest)+" "+ str(sws_start)+" " + str(sws_dur)+" " +str(ex_w)+" "+str(ex_in_w)+" " +str(ex_s)+" "+str(seed) + " " +init +" "+  date_i+  " "+ model_name + " " +str(NE+NI)
                                 
                                    


                                        lr_f = th_p_s + " " + th_d_s + " " + gp_s + " " + gd_s + " " + tau_pre_s + " " + tau_post_s + " " + sig_s + " " + tau_s_s + " " + lr + " "+ lr_p +" "
                                        print(pre_fix + lr_f + post_fix_w)
                                        
                                        stim = "stim{}_{}_{}hz_delay{}_{}_{}_{}_{}_{}".format(stim_start, stim_dur, stim_hz, delay, peak, dur, fy, ly, vrest)
                                        sws = "sws{}_{}_{}".format(sws_start, sws_dur, ex_s)
                                        
                                     
                                        
                                        if do_cargs==True:
                                            subprocess.run([pre_fix + lr_f + post_fix_w], shell=True)
                                            f0.write(pre_fix + lr_f + post_fix_w+'\n')
                                        
                                        if do_cargs==True:
                                            py_com = "python3 analysis_net_stimulation.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} &".format(lr, lr_p, model_name, date_i,th_p_s,th_d_s,gp_s, gd_s,tau_pre_s, tau_post_s,sig_s, tau_s_s, rho_ini,smin_ampa,smax_ampa,NE, Ip, Gex, con_M, con_M_in, con_ex_ex, con_ex_in, con_in_ex, con_in_in, T,Tp,dt,stim_start,stim_dur,stim_hz,delay,peak, dur,fy,ly,vrest, sws_start,sws_dur,ex_w,ex_in_w,ex_s, seed, init,con_log)
                                            subprocess.run([py_com], shell=True)
                                            subprocess.run(['echo start analyze!\n'], shell=True)
                                        else:
                                            py_com = "python3 analysis_net_stimulation.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(lr, lr_p, model_name, date_i,th_p_s,th_d_s,gp_s, gd_s,tau_pre_s, tau_post_s,sig_s, tau_s_s, rho_ini,smin_ampa,smax_ampa,NE, Ip, Gex, con_M, con_M_in, con_ex_ex, con_ex_in, con_in_ex, con_in_in, T,Tp,dt,stim_start,stim_dur,stim_hz,delay,peak, dur,fy,ly,vrest, sws_start,sws_dur,ex_w,ex_in_w,ex_s, seed, init,con_log)
                                            subprocess.run([py_com], shell=True)
                                            

                                        f0.write(py_com+'\n') 
                                        f0.write("echo analysis start"+'\n') 
                                        print(py_com)                          
f0.write("wait\n")
f0.close()




print("sh {}".format(sh_f_w))



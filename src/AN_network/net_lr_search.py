
import os
import sys
import matplotlib.pyplot as plt
from scipy import integrate
import math
from scipy.integrate import odeint
from multiprocessing import Process
import multiprocessing
from numba import jit,  njit, i8, f8
import numpy as np
import pandas as pd
from scipy.signal import periodogram, find_peaks
from scipy import signal

import matplotlib.pyplot as plt



args = sys.argv

lr_li = ["Hebbian", "Anti-Hebbian", "STDP", "Anti-STDP"]

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

con_M = float(args[c]) 
c+=1
con_M_in =  float(args[c])
c+=1


cpre_r = 1.0
cpost_r =1.0

con_ex_ex = float(args[c])
c+=1
con_ex_in = float(args[c])
c+=1
con_in_in = float(args[c])
c+=1
con_in_ex =float(args[c])
c+=1

T=int(args[c])
c+=1
Tp=int(args[c])
c+=1
# dt = 0.02
con_log = True
gabaup =False

seed = int(args[c])
c+=1

init =(args[c])
c+=1


ex_w_ori = float(args[c])
c+=1

bifur = args[c]
c+=1

param_n = args[c]
c+=1

print("search_bifur", bifur)
print("ex_w_ori", ex_w_ori)

NI = int(NE*Ip/(100-Ip))
print("NI", NI)

ex_diff=0

amp=float(lr_p[1:4])
tau = int(lr_p[5:7])
th=float(lr_p[9:12])
# amp = 0.7
# tau = 50 
# th = 0.6
offset =1000
T_w = offset + 3000
tbs=int(T/Tp)

con_log = True
gabaup = False


fs = 25
nowdir = os.getcwd()

rho_star = 0.5
zeta = 1
area =0.02
beta = 0.5
b=  5
g_nmdar = 0.0138403
g_cav_s = 0.050580

a_ca = 0.5
x_a = 34.8
x_tau = 0.2
s_a = 5
s_tau = 10
Vca = 120     #  [mV]
vrest = -70  #[mV]



@njit(i8[:](f8[:], i8,i8,f8,f8[:], f8[:],f8,f8,i8,i8, f8,f8,f8,f8,i8))
def Solvers(init, Lt, offset, dt, mp_pre, mp_post, cpre_r, cpost_r, sum_p, sum_d, th_p, th_d, tau_ca_pre, tau_ca_post, solver):

    
    rho_star = 0.5
    zeta = 1
    area=0.02
    beta = 0.5
    b=  5
    g_nmdar = 0.0138403
    g_cav_s = 0.050580

    a_ca = 0.5
    x_a = 34.8
    x_tau = 0.2
    s_a = 5
    s_tau = 10
    Vca = 120     # spike threshold [mV]
    vrest = -70 # reset potential [mV]

    
    def dAlldt(init, mp_pre, mp_post, it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post):
        

        x = init[0]
        s = init[1]
        cpre = init[2]
        cpost = init[3]
        ca = cpre + cpost
        nmda_f = 1.0 / (1.0 + np.exp(-(mp_pre[it]-20)/2.0))
        
        dxdt = (x_a * nmda_f - x/x_tau)
        dsdt = (s_a * x * (1-s) - s/s_tau)
        i_nmda = (g_nmdar * s * (mp_post[it]-Vca))*cpre_r
        
        inf_m=1.0 / (1.0 + np.exp(-(mp_post[it]+20)/9))
        
        i_vdcc_s = (g_cav_s * (inf_m)**2 * (mp_post[it]-Vca))*cpost_r
        dcpre = (- (a_ca * (i_nmda))- cpre/tau_ca_pre)
        dcpost = (- (a_ca * (10*area*i_vdcc_s)) - cpost/tau_ca_post)
#         dcadt =  dcpre + dcpost 
        ode = np.array([dxdt,dsdt, dcpre, dcpost])
        return ode
    
    for it in range(Lt):
        if it==0:
            continue
        cpre = init[2]
        cpost = init[3]
        ca = cpre + cpost


        if ca > th_p and it >= offset/dt:
            C_th_p = 1
            sum_p += 1
        else:
            C_th_p=0

        if ca > th_d and it >= offset/dt:
            C_th_d = 1
            sum_d += 1
        else:
            C_th_d = 0
        # 4th order Runge-Kutta 法
        if solver == 1:
            k1 = dt*dAlldt(init, mp_pre, mp_post, it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
            k2 = dt*dAlldt(init + 0.5*k1, mp_pre, mp_post, it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
            k3 = dt*dAlldt(init + 0.5*k2, mp_pre, mp_post,it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
            k4 = dt*dAlldt(init + k3, mp_pre, mp_post, it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
            init = init + (k1 + 2*k2 + 2*k3 + k4) / 6
            # 陽的 Euler 法
        elif solver == 0:
            init = init + dt*dAlldt(init, mp_pre, mp_post, it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
    
    return np.array([sum_p, sum_d])

class area_gd:
    def __init__(self, pars, T, offset, dt, cpre_r, cpost_r, solvers = 0):
        self.T = T
        self.offset = offset
        self.dt = dt
        self.presum_p = 0
        self.presum_d = 0
        self.postsum_p = 0
        self.postsum_d = 0 
        self.init = np.array([0.01,0.01,0,0])
        self.tc = 0
        self.th_p = pars["th_p"]
        self.th_d = pars["th_d"]
        self.gp= pars["gp"]
        self.gd= pars["gd"]
        self.tau_ca_pre =    pars["tau_ca_pre"] 
        self.tau_ca_post =   pars["tau_ca_post"]
        self.sig= pars["sig"] 
        self.tau_s = pars["tau_s"]
        self.pars = pars
        self.sol_num = solvers
        self.area=0.02
        self.g_nmdar = 0.0138403
        self.g_cav_s = 0.050580
        self.a_ca = 0.5
        self.x_a = 34.8
        self.x_tau = 0.2
        self.s_a = 5
        self.s_tau = 10
        self.Vca = 120     # spike threshold [mV]
        self.vrest = -70 
        self.cpre_r = cpre_r
        self.cpost_r = cpost_r

    def dAlldt_area(self, init, mp_pre, mp_post, it):

        x = init[0]
        s = init[1]
        cpre = init[2]
        cpost = init[3]
       # print("cpost", cpost.shape)
        ca = cpre + cpost#init[4]
    #     print("mppres", mp_pres[:,it].shape)
    #     print("x", x.shape)
        dxdt = (self.x_a * self.nmda_f0(mp_pre[it]) - x/self.x_tau)
        dsdt = (self.s_a * x * (1-s) - s/self.s_tau)

        i_nmda = (self.g_nmdar * s * (mp_post[it]-self.Vca))*self.cpre_r
        i_vdcc_s = (self.g_cav_s * (self.inf_m0(mp_post[it]))**2 * (mp_post[it]-self.Vca))*self.cpost_r

        dcpre = (- (self.a_ca * (i_nmda))- cpre/self.tau_ca_pre)
        dcpost = (- (self.a_ca * (self.area*10*i_vdcc_s)) - cpost/self.tau_ca_post)
       # dcpost = np.tile(dcpost, NE).reshape(-1,1)
#         dcadt =  dcpre + dcpost #(- a_ca * (i_nmda + i_vdcc_s) - ca/tau_ca_syn)
   
        ode = np.array([dxdt,dsdt, dcpre, dcpost])
        return ode
    
    def Solvers_area(self, func, init,  mp_pre, mp_post,it, solver=0):
        cpre = init[2]
        cpost = init[3]
        if cpre  > self.th_p and it > self.offset/self.dt:
            self.presum_p +=1
        if cpre  > self.th_d and it > self.offset/self.dt:
            self.presum_d +=1
        if cpost  > self.th_p and it > self.offset/self.dt: 
            self.postsum_p +=1
        if cpost  > self.th_d and it > self.offset/self.dt:
            self.postsum_d +=1
        # 4th order Runge-Kutta 法
        if solver == 1:
            k1 = dt*func(init, mp_pre, mp_post, it)
            k2 = dt*func(init + 0.5*k1, mp_pre, mp_post, it)
            k3 = dt*func(init + 0.5*k2, mp_pre, mp_post,it)
            k4 = dt*func(init + k3, mp_pre, mp_post, it)
            return init + (k1 + 2*k2 + 2*k3 + k4) / 6
            # 陽的 Euler 法
        elif solver == 0:
            return init + dt*func(init, mp_pre, mp_post, it)
        else:
            return None
    
    def run_ode_np_area(self, mp_pre, mp_post):
        Lt = int((self.T)/self.dt)
#         print("Lt",Lt)


        X_arr = np.zeros((4, Lt))
        X_arr[:,0] = self.init

        for it in range(Lt):

            init = self.Solvers_area(self.dAlldt_area, self.init,  mp_pre, mp_post, it, self.sol_num)
            self.init = init
            #print("init", init.shape)
            X_arr[:,it] = init

        init_t = int(self.offset/self.dt)

        cpre_amp = np.max(X_arr[2][init_t:])
        cpost_amp = np.max(X_arr[3][init_t:])

        if (cpre_amp > self.th_p and cpre_amp > self.th_d) or (cpost_amp > self.th_p and cpost_amp > self.th_d):
            gr = (self.postsum_d+self.presum_d)/(self.postsum_p+self.presum_p)
          #  print("gr", gr)
            gd = self.gp/gr
           # print("gd", gd)
        else:
            gd=self.gd
        
        self.pars["gd"] = gd
        
        
        
        return self.pars, X_arr[2,:], X_arr[3,:]
        
    def nmda_f0(self, v):
        return 1.0 / (1.0 + np.exp(-(v-20)/2.0))

    def inf_m0(self, v):
        return 1.0 / (1.0 + np.exp(-(v+20)/9))

def run_ode_lr(lag, T, offset, dt, cpre_r, cpost_r, mp_pre, mp_post,  syn_c_dic, gp, gd, th_p, th_d, tau_ca_pre, tau_ca_post, sig, taus,vrest):

    Lt=int((T)/dt)
    
    init_t = 0
    cpre = 0
    cpost = 0
#     ca = 0
    x_n= 0.01
    s_n = 0.01
    
    init = np.array([x_n,s_n,cpre,cpost])
    
    sum_p = 0
    sum_d = 0
    
   
    sums = Solvers(init, Lt, offset, dt, mp_pre, mp_post, cpre_r, cpost_r, 
                                  sum_p, sum_d,th_p, th_d, tau_ca_pre, tau_ca_post, 1)
    sum_p = sums[0]
    sum_d = sums[1]

    sec = (T-offset)/1000
#     print("sec", sec)
    ap = sum_p*(60/sec)/(60000/dt)
    ad = sum_d*(60/sec)/(60000/dt)

    if gp*ap+gd*ad==0:
        rho_b = 0.5
        syn_c = 1.0
    
    else:
        rho_b = gp*ap/(gp*ap+gd*ad)
        sig_r = np.sqrt(sig*sig*(ap+ad)/(gp*ap+gd*ad))
        tau_eff = (tau_s/dt)/(gp*ap+gd*ad)
        sig_r0 = 0
        
        u= U_p(0,rho_b,rho_star,sig_r,tau_eff,60000/dt)
        d = D_p(1,rho_b,rho_star,sig_r,tau_eff,60000/dt)

        
        syn_c = ((1-u)*beta+d*(1-beta)+b*(u*beta+(1-d)*(1-beta)))/(beta+(1-beta)*b)
#         syn_c0 = ((1-u0)*beta+d0*(1-beta)+b*(u0*beta+(1-d0)*(1-beta)))/(beta+(1-beta)*b)
        if syn_c<0:    
            syn_c=0

    syn_c_dic[int(lag)] = syn_c
#     print("lag {}, syn_c {}".format(lag, syn_c))
    return syn_c, cpre, cpost #, syn_c0        

def erf(x):
    y=lambda t: math.exp(-t*t)
    integ = integrate.quad(y, 0, x)
    v =2/np.sqrt(np.pi)*integ[0]
 
    return v

def U_p(rho_0,rho_b,rho_star,sig_r,tau_eff,T):
    x=(rho_star-rho_b+(rho_b-rho_0)*np.exp(-T/tau_eff))/(np.sqrt(sig_r*sig_r*(1-np.exp(-2*T*tau_eff))))
    eul = erf(-x)
    u= 1/2*(1+eul)
   
    return u

def D_p(rho_0,rho_b,rho_star,sig_r,tau_eff,T):
    x=(rho_star-rho_b+(rho_b-rho_0)*np.exp(-T/tau_eff))/(np.sqrt(sig_r*sig_r*(1-np.exp(-2*T/tau_eff))))
    eul = erf(-x)
    d= 1/2*(1-eul)
  
    return d



def fitting_curve(lr_li, lag_li, lag_step, syn_c_sorted, amp, tau,dx_li):
    esqs = np.zeros(4)
    dxs = np.zeros(4)
    
    esqs[0], dxs[0] = gaus_fit(lag_li, syn_c_sorted, amp, amp, tau,dx_li)
    esqs[1], dxs[1] = gaus_fit(lag_li, syn_c_sorted, -amp, -amp, tau,dx_li)
    esqs[2], dxs[2] = gaus_fit(lag_li, syn_c_sorted, -amp, amp, tau,dx_li)
    esqs[3], dxs[3] = gaus_fit(lag_li, syn_c_sorted, amp, -amp, tau,dx_li)
    
    min_esq = np.min(esqs)
   # print("min_residual", min_esq)
    min_dx = dxs[np.argmin(esqs)]
    lr_index = np.argmin(esqs)
    #print(lr_li[lr_index])
    return lr_index, min_esq, min_dx

def gaus_fit(lag_li, syn_c_sorted, a, b, tau, dx_li):
    esq_li = []
    for dx in dx_li:
    
        xl = np.arange(lag_li[0], -dx*lag_step, lag_step)
        xr = np.arange(-dx*lag_step, lag_li[-1]+lag_step/2, lag_step)
    
        yl = a*np.exp(-(xl+dx*lag_step)**2/tau**2)
        yr = b*np.exp(-(xr+dx*lag_step)**2/tau**2)
    
        y=np.append(yl, yr)

        y=y+1
        esq = np.sum((syn_c_sorted-y)*(syn_c_sorted-y))
        esq_li.append(esq)
    min_esq = np.min(esq_li)
    min_dx = dx_li[np.argmin(esq_li)]
    return min_esq, min_dx


def repro_wave(lr_index, lag_li, sync_li, lag_step, amp, tau, dx, gauss=True):
    xl = np.arange(lag_li[0], -dx*lag_step, lag_step)
    xr = np.arange(-dx*lag_step, lag_li[-1]+lag_step/2, lag_step)
    if lr_index == 0:
        a = amp
        b = amp
        yl = a*np.exp(-(xl+dx*lag_step)**2/tau**2)
        yr = b*np.exp(-(xr+dx*lag_step)**2/tau**2)
    if lr_index == 1:
        a = -amp
        b= -amp
        yl = a*np.exp(-(xl+dx*lag_step)**2/tau**2)
        yr = b*np.exp(-(xr+dx*lag_step)**2/tau**2)
    
    if lr_index == 2: 
        a = -amp
        b= amp
        yl = a*np.exp(-(xl+dx*lag_step)**2/tau**2)
        yr = b*np.exp(-(xr+dx*lag_step)**2/tau**2)
    
    
    if lr_index == 3:
        a = amp
        b= -amp
        yl = a*np.exp(-(xl+dx*lag_step)**2/tau**2)
        yr = b*np.exp(-(xr+dx*lag_step)**2/tau**2)

    else:
        yl = a*np.exp(-(xl+dx*lag_step)**2/tau**2)
        yr = b*np.exp(-(xr+dx*lag_step)**2/tau**2)
    y=np.append(yl, yr)
    y=y+1


    plt.figure(figsize=(8, 8))
    plt.plot(lag_li, sync_li) #label = "data")
    
    if gauss==True:
        plt.plot(lag_li, y, linestyle = "dashed") #, label="fitted")
   
    fs = 20
    fs_l= 15
    plt.xlabel('Time lag(ms)',fontsize = fs)
    plt.ylabel("Synaptic change", fontsize = fs)
    plt.xlim(-150, 150)
    plt.ylim(0.2, 1.8)
    yc=np.ones(len(lag_li))*1.0
    plt.plot(lag_li, yc, color="gray", alpha = 0.6)
    plt.xticks(fontsize = fs_l)
    plt.yticks(fontsize = fs_l)

    plt.show()
    
def random_samp(relm):
    x=np.random.rand(1)*(relm[1]-relm[0])+relm[0]
    return x[0]

def set_lr_params(th_p_r, th_d_r, gp_r, gd_r, tau_ca_pre_r, tau_ca_post_r, sig_r, tau_s_r):
    th_p=random_samp(th_p_r)
    th_d= random_samp(th_d_r)
    gp=random_samp(gp_r)
    gd=random_samp(gd_r)

    tau_ca_pre = random_samp(tau_ca_pre_r)
    tau_ca_post = random_samp(tau_ca_post_r)
    sig =random_samp(sig_r)
    tau_s = random_samp(tau_s_r)
    
    
    pars = {}
    
    pars["th_p"]  = th_p
    pars["th_d"] = th_d
    pars["gp"] =  gp
    pars["gd"] = gd
    pars["tau_ca_pre"] = tau_ca_pre
    pars["tau_ca_post"] =tau_ca_post
    pars["sig"] = sig 
    pars["tau_s"] =tau_s 
    
    return pars

def fre_spike_i(v, T, dt):
    
    if np.any(np.isinf(v)) or np.any(np.isnan(v)):
        print("fre_spike: nan")
        return "Excluded",0, 0, np.zeros(int(T/dt), dtype="int8")
    else:
#        
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
        return "not calc", maxfre, len(peaks), peak_np
    



ex_w = f"{ex_w_ori:,.2f}" 

if ex_w =="-0.00":
    ex_w = "0.00"

    
skip_search = 0
if param_n=="ori":

    nolr_f = "nolr_date_"+lr+"_ex"+bifur+".txt"
    if os.path.exists(nolr_f):
        f=open(nolr_f, "r")

        for l in f:
    #         print(l)
            ps = l.split(" ")
    #         print(ps)
            date_i_pre=ps[0]
            ex_w_pre = (ps[1]).strip("\n")

            if date_i == date_i_pre and ex_w == ex_w_pre:
                skip_search=1
                break
else:
    nolr_f = "nolr_date_params_"+lr+"_ex_"+param_n+bifur+".txt"
    if os.path.exists(nolr_f):
        f = open(nolr_f, "r")
        for n, l in enumerate(f):
#             if n>0:
#                 continue
            ps = l.split(" ")
            date_i_pre=ps[0]
    #         

            param_pre = (ps[1])
    #       

            ex_w_pre = (ps[2]).strip("\n")
    #       


            if param_n=="log":
                if date_i==date_i_pre and con_ex_ex == float(param_pre)and ex_w == ex_w_pre:
                    skip_search==1
                    break
            elif param_n=="syn":
                if date_i == date_i_pre and con_M==float(param_pre) and ex_w == ex_w_pre:
                    skip_search=1
    #                 print(True)
                    break
            elif param_n=="N":
                if date_i == date_i_pre and NE+NI==int(param_pre) and ex_w == ex_w_pre:
                    skip_search==1
                    break

if skip_search ==0:
                                            
    if con_log==False:
        drc ="./con_cu_i/{}/param_{}/NE{}_NI{}/con_th{}_{}_{}_{}/seed_{}/init_{}/T_{}/Tp_{}/".format(model_name, date_i, NE,NI, con_ex_ex,con_ex_in,con_in_ex,con_in_in,   seed,init,  T, Tp)

    elif con_log==True:
        drc ="./con_cu_i/{}/param_{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/seed_{}/init_{}/T_{}/Tp_{}/".format(model_name, date_i, NE, NI,con_M, con_ex_ex,con_M_in, con_in_in,  seed, init, T, Tp)

    offset = 1000
    move=1000
    T_w=3000
    tbs=int(T/Tp)
    c = int(T/1000 -1)#int((T-T_w)/move+1)

    fre_cvs_li=[]
    fre_m_li = []
    for n in range(NE):
    #         if n!=11:
    #             continue
        fre_li = []
        for tb in range(tbs):
            v_file = drc+ "ex"+str(ex_w)+"_"+"ex_in"+str(ex_w)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
            f2= open(v_file, "rb")
            rectype = np.dtype(np.float64)
            v_tb = np.fromfile(f2, dtype=rectype).astype("float64")
            if tb==0:
                v=v_tb
            else:
                v=np.append(v, v_tb)  #5s data v
            f2.close()
        dt = round(T/len(v),4)

        for i in range(c):
            v_p=v[int(offset/dt)+int(i*move/dt):int(offset/dt)+int(i*move/dt)+int(1000/dt)]
            pattern, maxfre, sp_c, peak_np = fre_spike_i(v_p, int(T_w/dt), dt)
            fre_li.append(sp_c/(T_w/1000))
        fre_cvs_li.append(np.std(fre_li)/np.mean(fre_li))
        fre_m_li.append(np.mean(fre_li))


    Hz_in = np.where((np.array(fre_m_li)>1) & (np.array(fre_m_li)<12))[0]
    fre_cvs_hz=np.take(fre_cvs_li, Hz_in)
    N_hz = np.take(np.arange(0,NE), Hz_in)

    if len(fre_cvs_hz) !=0:
        N_in = N_hz[np.argmin(fre_cvs_hz)]
        print("N_in", N_in)
    else:
        N_in=0
        print("N_in", N_in)

   

    T_w = offset + T_w
    print("T_w", T_w)



    tbs=int(5000/Tp)


    for n in range(NE):
            if n!=N_in:
                continue
    #        fre_li = []
            for tb in range(tbs):
                v_file = drc+ "ex"+str(ex_w)+"_"+"ex_in"+str(ex_w)+"_"+"N"+str(n)+"_"+str(tb)+".bin"
                f2= open(v_file, "rb")
                rectype = np.dtype(np.float64)
                v_tb = np.fromfile(f2, dtype=rectype).astype("float64")
                if tb==0:
                    v=v_tb
                else:
                    v=np.append(v, v_tb) #5s data v
                f2.close()

    dt = round(5000/len(v),4)
    step = int(0.1/dt)  
    print("dt",dt)
    p=np.arange(0,int(5000/dt), step)  #0.1 step
    v2=np.take(v,p)

    # plt.figure(figsize=(15,4))
    # plt.plot(v2)
    # plt.show()
    print("v2", v2.shape)

    lr_p="a{}t{}th{}".format(amp,tau,th)
    dx_li =[0] #np.arange(-3,3,1, dtype = "int8") 
    # fname = date_i+"_"+bifur+"_"+lr_li[lr_in]+ "_"+lr_p +"_ex_1"


    # #Hebbian
    if lr == "Hebbian":
        th_d_r=[0.06, 1.6]
        th_p_r=[0.06, 1.6]
        gp_r = [50,5000]
        gd_r=[50,10000]

        tau_ca_pre_r = [1,100]
        tau_ca_post_r = [1,100]
        sig_r = [0.5, 35]
        tau_s_r = [2.5*1000, 2500*1000]


    elif lr ==  "Anti-Hebbian":
        th_p_r=[0.06, 1.6]
        th_d_r=[0.06, 1.6]
        gp_r = [50,5000]
        gd_r=[50,10000]

        tau_ca_pre_r = [1,100]
        tau_ca_post_r = [1,100]
        sig_r = [0.5, 35]
        tau_s_r = [2.5*1000, 2500*1000]

    elif lr== "STDP":  
    # #STDP
        th_p_r=[0.8, 1.6]
        th_d_r=[0.5, 1.0]
        gp_r = [50,5000]
        gd_r=[50,10000]
        tau_ca_pre_r = [1,100]
        tau_ca_post_r = [1,100]
        sig_r = [0.5, 35]
        tau_s_r = [2.5*1000, 2500*1000]



    elif lr == "Anti-STDP":
        th_d_r=[0.8, 1.6]
        th_p_r=[0.5, 1.0]
        gp_r = [50,5000]
        gd_r=[50,10000]
        tau_ca_pre_r = [1,100]
        tau_ca_post_r = [1,100]
        sig_r = [0.5, 35]
        tau_s_r = [2.5*1000, 2500*1000]


    dt = 0.1
    lag_step=10
    lag_li = np.arange(-100, 110, lag_step)



    #ms

    N=3000
    c=0
    d=0


    savedir ="./lr_params_con/{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}/ex{}/".format(model_name, date_i, NE,NI,con_M,con_ex_ex,con_M_in,con_in_in, ex_w_ori)

    try:
        os.makedirs(savedir)
    except FileExistsError:
        print("{} is already exist".format(savedir))


    for i in range(N):
            if i%1000 == 0:
                print("i", i)

            if __name__ == "__main__":

                pars = set_lr_params(th_p_r, th_d_r, gp_r, gd_r, tau_ca_pre_r, tau_ca_post_r, sig_r, tau_s_r)

                cpre_r,cpost_r = 1,1
                lag=100

                vpre = v2 #s_s[:, 0]
                vpost = (np.roll(vpre, int(lag/dt)))[0:int(T_w/dt)]
                vpre = vpre[0:int(T_w/dt)]


                area_class = area_gd(pars, T_w, offset, dt,cpre_r, cpost_r, 1)
                pars, cpre_t, cpost_t = area_class.run_ode_np_area(vpre, vpost)


                cpre_r = 0.7/np.max(cpre_t[int(offset/dt):])
                cpost_r = 1.4/np.max(cpost_t[int(offset/dt):])

                area_class2 = area_gd(pars,T_w, offset, dt,cpre_r, cpost_r,  1)
                pars, cpre_t, cpost_t = area_class2.run_ode_np_area(vpre, vpost)


                tau_ca_pre  = pars['tau_ca_pre']
                tau_ca_post  = pars['tau_ca_post']
                tau_s  = pars['tau_s'] 
                sig = pars['sig']
                th_d, th_p = pars['th_d'], pars['th_p']
                gd, gp = pars['gd'], pars['gp']
    #             print("th_p", th_p)
    #             print("th_d", th_d)


                if gd > gd_r[1]:
                    continue

                results=[]
                manager = multiprocessing.Manager()
                syn_c_dic = manager.dict()
    #             syn_c_dic={}
                for lag in lag_li:
                    vpre = v2 #s_s[:, 0]
                    vpost = (np.roll(vpre, int(lag/dt)))[0:int(T_w/dt)]
                    vpre = vpre[0:int(T_w/dt)]

                    fs=20
                    p = Process(target=run_ode_lr, args=(lag, T_w, offset, dt, cpre_r, cpost_r, vpre, vpost,  syn_c_dic,gp, gd, th_p, th_d, tau_ca_pre, tau_ca_post, sig, tau_s,vrest))


                    p.start()
                    results.append(p)

                for p in results:
                    p.join()

                a = syn_c_dic.items()
                #print("syncdic", a)
                new_dic = sorted(a)
                syn_c_sorted = np.array([i[1] for i in new_dic])

                lr_index, min_esq, min_dx = fitting_curve(lr_li, lag_li, lag_step, syn_c_sorted, amp, tau,dx_li)
    #             repro_wave(lr_index, lag_li, syn_c_sorted, lag_step, amp, tau, min_dx)


                pars_sv = {}
                pars_sv["th_p"]  = pars["th_p"] 
                pars_sv["th_d"] =pars["th_d"] 
                pars_sv["gp"] = pars["gp"]
                pars_sv["gd"] = pars["gd"]
                pars_sv["tau_ca_pre"] = pars["tau_ca_pre"]
                pars_sv["tau_ca_post"] = pars["tau_ca_post"]
                pars_sv["sig"] = pars["sig"]
                pars_sv["tau_s"] =  pars["tau_s"]
                pars_sv["lse"] =  min_esq
               # print(pars_sv)

                if min_esq > th:
                    lr_index = 4


                if lr_index!=4 and lr_li[lr_index] == lr:
                    repro_wave(lr_index, lag_li, syn_c_sorted, lag_step, amp, tau, min_dx)
                    print("Fitting!")
                    pars_sv["lr"] = lr_index
                    print(lr_li[lr_index])
    #                 print("min_esq", min_esq)
                    print(pars_sv)
                    pd_param = pd.DataFrame(pars_sv, index = [c])
                    print(pd_param)


                    c += 1
                    d += 1

                    print("fit count", d)
                    print("save params")


                if d==1:
                    break

    if d==1:
        pd_param.to_excel(savedir + lr+"_"+lr_p+".xlsx")   
        print("save: ", savedir + lr+"_"+lr_p+".xlsx")
    else:
        if param_n== "ori":
            print("{} lr not found".format(date_i))
            nolr_f = "nolr_date_"+lr+"_ex"+bifur+".txt"
            if not os.path.exists(nolr_f):
                f = open(nolr_f, "w")
            else:
                f = open(nolr_f, "a")
            f.write(date_i+" " + str(ex_w)+"\n")
            f.close()

        else:
            print("{} lr not found".format(date_i))
            nolr_f = "nolr_date_params_"+lr+"_ex_"+param_n+bifur+".txt"
            if not os.path.exists(nolr_f):
                f = open(nolr_f, "w")
            else:
                f = open(nolr_f, "a")
            if param_n=="log":
                f.write(date_i+" " +str(con_ex_ex)+" "+ str(ex_w)+"\n")
            elif param_n=="syn":
                f.write(date_i+" " + str(con_M)+" "+str(ex_w)+"\n")
            elif param_n=="N":
                f.write(date_i+" " + str(int(NE+NI))+" "+str(ex_w)+"\n")
            f.close()
else:
    print("search skip because of nolr")


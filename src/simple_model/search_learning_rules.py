import os
import matplotlib.pyplot as plt
from scipy import integrate
import math
from multiprocessing import Process
import multiprocessing
from scipy import interpolate
from numba import jit,  i8, f8
import numpy as np
from datetime import datetime
import pandas as pd

#python3 search_learning_rules.py

#fitting curve parameters
amp = 0.7
tau = 50 
th = 0.25
lr_p="a{}t{}th{}".format(amp,tau,th)

#learning rule index
lr_in = 0

lr_li = ["Hebbian", "Anti-Hebbian", "STDP", "Anti-STDP"]

dx_li =[0] 

fname = lr_li[lr_in]+ "_"+lr_p 


rho_star = 0.5
zeta = 1

beta = 0.5
b=  5
g_nmdar = 0.0138403
g_cav_s = 0.050580
a_ca = 0.5
x_a = 34.8
x_tau = 0.2
s_a = 5
s_tau = 10
Vca = 120    
vrest = -70 
amp = 0.7
tau = 50 
th = 0.25

test_T=3000 #offset including
offset=2000
T = 3000 #offset including
dt = 0.1
lag_step=10
lag_li = np.arange(-160, 170, lag_step)

peak =13  
fx,fy = int(-0.8/dt),-68  #down to up
lx,ly = int(0.8/dt), -76  #down
vrest = -70

off = 200 #ms 
n=3
interval = 1000 #ms
stim_T = n *interval

N=300000
c=0
# d=0
 
nowdir = os.getcwd()
# now = datetime.now()
# date = f"{now.year}_{now.month}_{now.day}" #"2022_7_23"
savedir = nowdir 
try:
    os.makedirs(savedir)
except FileExistsError:
    print("{} is already exist".format(savedir))


th_d_r=[0.06, 1.6]
th_p_r=[0.06, 1.6]
gp_r = [50,5000]
gd_r=[50,10000]
tau_ca_pre_r = [1,100]
tau_ca_post_r = [1,100]
sig_r = [0.5, 35]
tau_s_r = [2.5*1000, 2500*1000]



@jit(i8[:](f8[:], i8,i8,f8,f8[:], f8[:],f8,f8,i8,i8, f8,f8,f8,f8,i8))
def Solvers(init, Lt, offset, dt, mp_pre, mp_post, cpre_r, cpost_r, sum_p, sum_d, th_p, th_d, tau_ca_pre, tau_ca_post, solver):
    th_p = th_p 
    th_d = th_d 
    tau_ca_pre=tau_ca_pre 
    tau_ca_post= tau_ca_post
    
    rho_star = 0.5
    zeta = 1

    beta = 0.5
    b=  5
    g_nmdar = 0.0138403
    g_cav_s = 0.050580

    a_ca = 0.5
    x_a = 34.8
    x_tau = 0.2
    s_a = 5
    s_tau = 10
    Vca = 120     
    vrest = -70 

    
    
        
    
    def dAlldt(init, mp_pre, mp_post, it, cpre_r, cpost_r):

        x = init[0]
        s = init[1]
        cpre = init[2]
        cpost = init[3]
        ca = cpre + cpost
        nmda_f = 1.0 / (1.0 + np.exp(-(mp_pre[it]-20)/2.0))
        
        dxdt = (x_a * nmda_f - x/x_tau)
        dsdt = (s_a * x * (1-s) - s/s_tau)
        i_nmda = (g_nmdar * s * (vrest-Vca))*cpre_r
        
        inf_m=1.0 / (1.0 + np.exp(-(mp_post[it]+20)/9))
        
        i_vdcc_s = (g_cav_s * (inf_m)**2 * (mp_post[it]-Vca))*cpost_r
        dcpre = (- (a_ca * (i_nmda))- cpre/tau_ca_pre)
        dcpost = (- (a_ca * (i_vdcc_s)) - cpost/tau_ca_post)
        dcadt =  dcpre + dcpost 
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
        # 4th order Runge-Kutta 
        if solver == 1:
            k1 = dt*dAlldt(init, mp_pre, mp_post, it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
            k2 = dt*dAlldt(init + 0.5*k1, mp_pre, mp_post, it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
            k3 = dt*dAlldt(init + 0.5*k2, mp_pre, mp_post,it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
            k4 = dt*dAlldt(init + k3, mp_pre, mp_post, it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
            init = init + (k1 + 2*k2 + 2*k3 + k4) / 6
            # Euler 
        elif solver == 0:
            init = init + dt*dAlldt(init, mp_pre, mp_post, it, cpre_r, cpost_r, th_p, th_d, tau_ca_pre, tau_ca_post)
    
    return np.array([sum_p, sum_d])




# simple 1Hz spikes
def gen_spikes(T, off, dt, n, interval,lag):

        spikes= np.zeros((2,int((T)/dt)))
        
        for i in range(n):
          
            if int(off/dt)+i*int(interval/dt) < int((T+off)/dt):
                spikes[0][int(off/dt)+i*int(interval/dt)] = 1
            if int(off/dt)+i*int(interval/dt) + int(lag/dt) < int((T+off)/dt):
                spikes[1][int(off/dt)+i*int(interval/dt) + int(lag/dt)]=1
       
        return spikes

#convert spike trains to membrane potential by linear interpolation   
def sp_to_mp(spikes, T, off ,peak, fx,fy, lx,ly, vrest):
    Lt = int((T)/dt)
    pre_x = np.arange(0,Lt,1)* spikes[0]
    pre_y = spikes[0]*peak
    post_x = np.arange(0,Lt,1)* spikes[1]
    post_y = spikes[1]*peak
    for m,j in enumerate(spikes[0]):
        if j==1:
          
                pre_x[m+lx]=pre_x[m] + lx
                pre_y[m+lx]= ly
        
                pre_x[m+fx]=pre_x[m] + fx
                pre_y[m+fx]= fy

    for m,j in enumerate(spikes[1]):
        if j==1:
    
                post_x[m+lx]=post_x[m] + lx
                post_y[m+lx]= ly
           
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
    
     
    ip2 = ["liner interpolation", lambda x, y: interpolate.interp1d(x,y,fill_value="extrapolate")]
    for method_name, method in [ip2]:
    
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
        self.cpre_r = cpre_r
        self.cpost_r = cpost_r

    def dAlldt_area(self, init, mp_pre, mp_post, it):

        x = init[0]
        s = init[1]
        cpre = init[2]
        cpost = init[3]
       # print("cpost", cpost.shape)
        ca = cpre + cpost#init[4]

        dxdt = (x_a * self.nmda_f0(mp_pre[it]) - x/x_tau)
        dsdt = (s_a * x * (1-s) - s/s_tau)

        i_nmda = (g_nmdar * s * (vrest-120))*self.cpre_r
        i_vdcc_s = (g_cav_s * (self.inf_m0(mp_post[it]))**2 * (mp_post[it]-Vca))*self.cpost_r

        dcpre = (- (a_ca * (i_nmda))- cpre/self.tau_ca_pre)
        dcpost = (- (a_ca * (i_vdcc_s)) - cpost/self.tau_ca_post)
 
        dcadt =  dcpre + dcpost 
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
        # 4th order Runge-Kutta 
        if solver == 1:
            k1 = dt*func(init, mp_pre, mp_post, it)
            k2 = dt*func(init + 0.5*k1, mp_pre, mp_post, it)
            k3 = dt*func(init + 0.5*k2, mp_pre, mp_post,it)
            k4 = dt*func(init + k3, mp_pre, mp_post, it)
            return init + (k1 + 2*k2 + 2*k3 + k4) / 6
            # Euler 
        elif solver == 0:
            return init + dt*func(init, mp_pre, mp_post, it)
        else:
            return None
    
    def run_ode_np_area(self, mp_pre, mp_post):
        Lt = int((self.T)/self.dt)


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

        try:
            if (cpre_amp > self.th_p and cpre_amp > self.th_d) or (cpost_amp > self.th_p and cpost_amp > self.th_d):
                gr = (self.postsum_d+self.presum_d)/(self.postsum_p+self.presum_p)
                gd = self.gp/gr
       
            else:
                gd=self.gd

            self.pars["gd"] = gd
        except:
            gd=0

        
        
        return self.pars, X_arr[2,:], X_arr[3,:]
        
    def nmda_f0(self, v):
        return 1.0 / (1.0 + np.exp(-(v-20)/2.0))

    def inf_m0(self, v):
        return 1.0 / (1.0 + np.exp(-(v+20)/9))

def run_ode_lr(lag, T, offset, dt, cpre_r, cpost_r, mp_pre, mp_post,  syn_c_dic, gp, gd, th_p, th_d, tau_ca_pre, tau_ca_post, sig, taus):

    

    Lt=int((T)/dt)
    
    peak =13  #12.692
    fx,fy = int(-0.8/dt),-68  #down to up
    lx,ly = int(0.8/dt), -76  #down
    vrest = -70

    off = 200 #ms 
    n=3
    interval = 1000 #ms
    stim_T = n *interval
    
    spikes_stim = gen_spikes(stim_T,off,dt,n,interval,lag)
    mp_pre, mp_post =sp_to_mp(spikes_stim, stim_T,off, peak, fx, fy, lx, ly, vrest)
  

    init_t = 0
    cpre = 0
    cpost = 0
    x_n= 0.01
    s_n = 0.01
    
    init = np.array([x_n,s_n,cpre,cpost])
    
    sum_p = 0
    sum_d = 0
    
   
    sums = Solvers(init, Lt, offset, dt, mp_pre, mp_post, cpre_r, cpost_r, 
                                  sum_p, sum_d,th_p, th_d, tau_ca_pre, tau_ca_post, 1)
    sum_p = sums[0]
    sum_d = sums[1]
    ap = sum_p*60/(60000/dt)
    ad = sum_d*60/(60000/dt)

    
    if gp*ap+gd*ad==0:
        rho_b = 0.5
        syn_c = 1.0
    
    else:
        rho_b = gp*ap/(gp*ap+gd*ad)
        sig_r = np.sqrt(sig*sig*(ap+ad)/(gp*ap+gd*ad))
        tau_eff = (tau_s/dt)/(gp*ap+gd*ad)
        sig_r0 = 0
        
        u= U_p(0,rho_b,rho_star,sig_r,tau_eff,60,interval/dt)
        d = D_p(1,rho_b,rho_star,sig_r,tau_eff,60,interval/dt)
        
        syn_c = ((1-u)*beta+d*(1-beta)+b*(u*beta+(1-d)*(1-beta)))/(beta+(1-beta)*b)
        if syn_c<0:    
            syn_c=0

            
    syn_c_dic[int(lag)] = syn_c
    

    return syn_c, cpre, cpost    
            


def erf(x):
    y=lambda t: math.exp(-t*t)
    integ = integrate.quad(y, 0, x)
    v =2/np.sqrt(np.pi)*integ[0]
    return v

def U_p(rho_0,rho_b,rho_star,sig_r,tau_eff,n,interval):
    x=(rho_star-rho_b+(rho_b-rho_0)*np.exp(-n*interval/tau_eff))/(np.sqrt(sig_r*sig_r*(1-np.exp(-2*n*interval/tau_eff))))
    eul = erf(-x)
    u= 1/2*(1+eul)
    return u

def D_p(rho_0,rho_b,rho_star,sig_r,tau_eff,n,interval):
    x=(rho_star-rho_b+(rho_b-rho_0)*np.exp(-n*interval/tau_eff))/(np.sqrt(sig_r*sig_r*(1-np.exp(-2*n*interval/tau_eff))))
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
    min_dx = dxs[np.argmin(esqs)]
    lr_index = np.argmin(esqs)

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


pd_all=[]
for i in range(N):
        if i%100 == 0:
            print("i", i)
        
        if __name__ == "__main__":

            pars = set_lr_params(th_p_r, th_d_r, gp_r, gd_r, tau_ca_pre_r, tau_ca_post_r, sig_r, tau_s_r)
            
            if lr_in ==0 or lr_in ==2:
                if pars["th_p"] < pars["th_d"]:
                    continue
            else:
                if pars["th_p"] > pars["th_d"]:
                    continue
    
            cpre_r,cpost_r = 1,1
            
            #update pars["gd"] for NO change in large dfferences 
            lag=100
            spikes_stim = gen_spikes(stim_T,off,dt,n,interval,lag)
            vpre, vpost =sp_to_mp(spikes_stim, stim_T, off, peak, fx, fy, lx, ly, vrest)
            area_class = area_gd(pars, test_T, offset, dt,cpre_r, cpost_r, 1)
            pars, cpre_t, cpost_t = area_class.run_ode_np_area(vpre, vpost)
            
            #to convert amplitudes of calcium from NMDAR and VGCC to 0.7 and 1.4   
            cpre_r = 0.7/np.max(cpre_t[int(offset/dt):int(test_T/dt)])
            cpost_r = 1.4/np.max(cpost_t[int(offset/dt):int(test_T/dt)])

            #update pars["gd"] again
            area_class2 = area_gd(pars,test_T, offset, dt,cpre_r, cpost_r,  1)
            pars, cpre_t, cpost_t = area_class2.run_ode_np_area(vpre, vpost)
            tau_ca_pre  = pars['tau_ca_pre']
            tau_ca_post  = pars['tau_ca_post']
            tau_s  = pars['tau_s'] 
            sig = pars['sig']
            th_d, th_p = pars['th_d'], pars['th_p']
            gd, gp = pars['gd'], pars['gp']
    
            
          
            if gd > gd_r[1] or gd < gd_r[0]:
                continue
            
                
            results=[]
            manager = multiprocessing.Manager()
            syn_c_dic = manager.dict()
            for lag in lag_li:
                fs=20
                p = Process(target=run_ode_lr, args=(lag, T, offset, dt, cpre_r, cpost_r, vpre, vpost,  syn_c_dic,gp, gd, th_p, th_d, tau_ca_pre, tau_ca_post, sig, tau_s))
#                
                   
                p.start()
                results.append(p)
                
            for p in results:
                p.join()
            
            a = syn_c_dic.items()
     
            new_dic = sorted(a)
            syn_c_sorted = np.array([i[1] for i in new_dic])

            lr_index, min_esq, min_dx = fitting_curve(lr_li, lag_li, lag_step, syn_c_sorted, amp, tau,dx_li)
#             
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
       

            if lr_index == lr_in and min_esq<th:
                repro_wave(lr_index, lag_li, syn_c_sorted, lag_step, amp, tau, min_dx)
                print("Fitting!")
                re_c=0
                pars_sv["lr"] = lr_index
                print(lr_li[lr_index])
#       
                print(pars_sv)
                pd_param = pd.DataFrame(pars_sv, index = [c])
                print(pd_param)
                if c==0:
                    pd_all.append(pd_param)

                else:
                    pd_all.append(pd_param)
                    pd_params = pd.concat(pd_all, ignore_index=True) 
           
                c += 1
                # d += 1
                
                print("fit count", c)
                print("save params")

            if c==999:
                break
            
pd_params.to_excel(savedir + "/" +fname+".xlsx"  )


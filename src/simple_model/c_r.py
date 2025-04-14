import os
import sys
from scipy import interpolate
import numpy as np

args = sys.argv


def gen_spikes(T, off, dt, n, interval,lag):

        spikes= np.zeros((2,int((T)/dt)), dtype=np.int8)

        for i in range(n):
           # print(i)
            
            spikes[0][int(off/dt)+i*int(interval/dt)] = 1
            spikes[1][int(off/dt)+i*int(interval/dt) + int(lag/dt)]=1
       # print(spikes.shape)
        return spikes

def sp_to_mp_2(spikes,Lt ,peak, fx,fy, lx,ly, vrest): 
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
    
    fitted_curve_pre = interpolate.interp1d(pre_x_a,pre_y_a,fill_value="extrapolate")
    fitted_curve_post = interpolate.interp1d(post_x_a,post_y_a,fill_value="extrapolate")
    
    middle_pre = fitted_curve_pre(x_latent_pre)
    middle_post = fitted_curve_post(x_latent_post)

    first_pre = np.ones(x_latent_pre[0])*vrest
    last_pre = np.ones(Lt-1-x_latent_pre[-1])*vrest
    first_post = np.ones(x_latent_post[0])*vrest
    last_post = np.ones(Lt-1-x_latent_post[-1])*vrest
    mp_pre = np.append(first_pre,middle_pre).astype("float32")
    mp_pre = np.append(mp_pre,last_pre).astype("float32")
    mp_post = np.append(first_post,middle_post).astype("float32")
    mp_post = np.append(mp_post,last_post).astype("float32")
   
    return mp_pre, mp_post

class area_gd:
    def __init__(self, T, offset, dt, cpre_r, cpost_r, th_p, th_d, gp, gd, tau_ca_pre, tau_ca_post, sig,tau_s, solvers = 0):
        self.T = T
        self.offset = offset
        self.dt = dt
        self.presum_p = 0
        self.presum_d = 0
        self.postsum_p = 0
        self.postsum_d = 0 
        self.init = np.array([0.01,0.01,0,0])
        self.tc = 0
        self.th_p = th_p
        self.th_d = th_d
        self.gp= gp
        self.gd= gd
        self.tau_ca_pre =    tau_ca_pre
        self.tau_ca_post =   tau_ca_post
        self.sig= sig
        self.tau_s = tau_s
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
    #     print("mppres", mp_pres[:,it].shape)
    #     print("x", x.shape)
        dxdt = (x_a * self.nmda_f0(mp_pre[it]) - x/x_tau)
        dsdt = (s_a * x * (1-s) - s/s_tau)

        i_nmda = (g_nmdar * s * (vrest-120))*self.cpre_r
        i_vdcc_s = (g_cav_s * (self.inf_m0(mp_post[it]))**2 * (mp_post[it]-Vca))*self.cpost_r

        dcpre = (- (a_ca * (i_nmda))- cpre/self.tau_ca_pre)
        dcpost = (- (a_ca * 10*area*(i_vdcc_s)) - cpost/self.tau_ca_post)
       # dcpost = np.tile(dcpost, NE).reshape(-1,1)
        dcadt =  dcpre + dcpost #(- a_ca * (i_nmda + i_vdcc_s) - ca/tau_ca_syn)
   
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
#         print("cpre_amp", cpre_amp)
#         print("cpost_amp", cpost_amp)
        
#         print("pre_sump",self.presum_p)
#         print("pre_sumd",self.presum_d)
#         print("post_sump",self.postsum_p)
#         print("post_sumd",self.postsum_d)
        if (cpre_amp > self.th_p and cpre_amp > self.th_d) or (cpost_amp > self.th_p and cpost_amp > self.th_d):
            gr = (self.postsum_d+self.presum_d)/(self.postsum_p+self.presum_p)
          #  print("gr", gr)
            gd = self.gp/gr
           # print("gd", gd)
        else:
            gd=self.gd
        
        
        return gd, X_arr[2,:], X_arr[3,:]
        
    def nmda_f0(self, v):
        return 1.0 / (1.0 + np.exp(-(v-20)/2.0))

    def inf_m0(self, v):
        return 1.0 / (1.0 + np.exp(-(v+20)/9))



if __name__ == "__main__":
    test_T=3000
    offset=2000
    # dt = 0.04

    th_p = float(args[1])
    th_d = float(args[2])
    gp= float(args[3])
    gd= float(args[4])
    tau_ca_pre=float(args[5])
    tau_ca_post= float(args[6])
    sig=float(args[7])
    tau_s=float(args[8])
    order = args[9] 
    dt = float(args[10])
    lr_n = args[11] 
    lr_p = args[12] 
    
    # print("lr_p", lr_p)
    
    a_ca = 0.5
    rho_star = 0.5
    zeta = 1
    g_nmdar = 0.0138403 # 8.8589 * 1.363290
    g_cav_s = 0.050580
    a_ca = 0.5
    x_a = 34.8 
    x_tau = 0.2
    s_a = 5 
    area=0.02
    s_tau = 10
    Vca = 120   # spike threshold [mV]
    vrest = -70

    lag=100   #ms
    off=200  #ms
    n=60
    interval = 1000 #ms
    stim_T = 60000 #ms
    Lt_stim:int = int(stim_T/dt)

    fxs,fys = int(-0.8/dt),-68
    lxs,lys = int(0.8/dt), -76
    peak =13  #12.692
  
    cpre_r, cpost_r = 1,1

    spikes_stim = gen_spikes(stim_T,off,dt,n,interval,lag)
    vpre, vpost =sp_to_mp_2(spikes_stim, Lt_stim, peak, fxs, fys, lxs, lys, vrest)
    area_class =  area_gd(test_T, offset, dt,cpre_r, cpost_r, th_p, th_d, gp, gd, tau_ca_pre, tau_ca_post, sig,tau_s, 1)
    gd, cpre_t, cpost_t = area_class.run_ode_np_area(vpre, vpost)
    cpre_r = 0.7/np.max(cpre_t[int(offset/dt):int(test_T/dt)])
    cpost_r = 1.4/np.max(cpost_t[int(offset/dt):int(test_T/dt)])
    # print("cpre_r", cpre_r)
    # print("cpost_r", cpost_r)
    area_class =  area_gd(test_T, offset, dt,cpre_r, cpost_r, th_p, th_d, gp, gd, tau_ca_pre, tau_ca_post, sig,tau_s, 1)
    gd, cpre_t, cpost_t = area_class.run_ode_np_area(vpre, vpost)
    
    savedir = os.getcwd()
    path = savedir +"/"+ "c_r_"+lr_n+"_"+lr_p+"_"+order+".txt"
    try:
        os.makedirs(savedir)
    except FileExistsError:
        print("{} is already exist".format(savedir))  
    with open(path, mode = "w")  as f:
        f.write(str(cpre_r)+"\n")
        f.write(str(cpost_r)+"\n")
        
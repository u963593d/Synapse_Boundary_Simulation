
import os
from scipy import interpolate
import gc
import numpy as np
import math 

# param_search
#parameters for spike search
NE = 10 #number of neurons
T = 360000 #ms
Tp = 600000 #ms
dt = 0.05  #step size
offset = 0  #ms
seed1=40

param = 1.3   #initial value for log10(mean of Inter-spike interval)
           
step=0.00002  #step size for chaging parameter 
lr = 0.97     #learning rate of step size for chaging parameter

fr_li =[10, 3, 1, 0.3, 0.1]  #list for mean firing rate
fr_name_li =["10.0","3.0","1.0", "0.30", "0.10"] #string list for mean firing rate
lim =[ 0.5, 0.5, 0.5, 0.05, 0.05] #precision for mean firing rates


#spike parameters obtained from linear regression analysis of vivo data
a_ISI = 0.03
b_ISI = 0.65

ISI_lim=-b_ISI/a_ISI
print("ISI_lim <", ISI_lim)

#parameters for conversion from spike data to votage waveforms
peak =13                          #voltage peak of spikes (mv)
fx,lx =  int(-0.8/dt),  int(0.8/dt)  #duration of action potentials  (ms)
fy, lyup = -61, -61                  #membrane potential of Up state (mv)
fyd, ly = -76, -76                   #membrane potential of Down state (mv)
vrest = -70                          #resting potential  (mv)



# functions
#generate spike trains
def Poisson_generator_w(n, T, dt, M, myseed=False):
  #print("M", M)
  Lts = int(T/dt)

  SD = a_ISI * M + b_ISI 
  if myseed:
    np.random.seed(seed=myseed)
  else:
    np.random.seed()
    

  # generate uniformly distributed random variables
  poisson_train = np.zeros((n,Lts)) 
  
  for i in range(n):
      t=0
      while (t < Lts):
         t += (10**(np.random.normal(M, SD, 1)[0]))/dt #ms
         if t <Lts:
             poisson_train[i][math.floor(t)] = 1

  return poisson_train


def gen_spikes_wake(N, T, dt, M, seed):
    spikes = Poisson_generator_w(N+1,T, dt, M, seed)
    return spikes

def sp_to_mp(spt, spt_ID, Lt ,peak, fx, fy,fyd, lx, ly, lyup, vrest): 
    
    x = np.arange(0,Lt,1)* spt
    y = spt*peak
   
    for m,j in enumerate(spt):
        if j==1:
            if spt_ID[m] == 0:    #First spike in burst 
                if m+lx < len(spt):
                    x[m+lx]=m + lx
                    y[m+lx]= lyup
                if m+fx >=0:
                    x[m+fx]= m + fx
                    y[m+fx]= fyd
            
            elif spt_ID[m] == -1:    #last spike in burst 
                if m+lx < len(spt):
                    x[m+lx]=m + lx
                    y[m+lx]= ly
                if m+fx >=0:
                    x[m+fx]= m + fx
                    y[m+fx]= fy
                    
            elif spt_ID[m] == -2:    #
                if m+lx < len(spt):
                    x[m+lx]=m + lx
                    y[m+lx]= ly
                if m+fx >=0:
                    x[m+fx]= m + fx
                    y[m+fx]= fyd
                    
            else:
                if m+lx < len(spt):
                    x[m+lx]=m + lx
                    y[m+lx]= lyup
                if m+fx >=0:
                    x[m+fx]= m + fx
                    y[m+fx]= fy

    x_a = np.take(x, np.nonzero(x)[0])
    y_a = np.take(y, np.nonzero(y)[0])
#     print("x_a",x_a)
#     print("y_a", y_a)
    if spt[0]==1:
        x_a = np.insert(x_a,0,0)
    #         print("xa",x_a)

    if x_a[0] != 0:         
        x_a = np.insert(x_a, 0, 0)
        y_a = np.insert(y_a, 0, fyd)

    if x_a[-1] != len(spt)-1:
        x_a = np.append(x_a, len(spt)-1)
        y_a = np.append(y_a, ly)

    x_latent = np.arange(min(x_a), max(x_a)+1, 1).astype("int32")

    fitted_curve = interpolate.interp1d(x_a,y_a,fill_value="extrapolate")
    middle = np.array((fitted_curve(x_latent)))
    return middle

def spike_search_w(fr_li, fr_name_li, lim,  NE, T, dt, offset, lr, step, param, seed1):
    Lt = int((T +offset)/dt) #0.1ms
    
    lim = [i/10 for i in lim]
    
    rate_li = []
    param_li=[]

    pc=0
    mc=0
    
        
    for n,fr in enumerate(fr_li):
        print("search:{} Hz".format(fr))

        while(1):
            print("param", param)

#             if param > ISI_w_lim:
#                 break
            spikes = gen_spikes_wake(NE, (T+offset),dt, param, seed1)

            av_rate = round(np.mean(np.sum(spikes, axis = 1)/((T+offset)/1000)), 4) #averaged rate
            print("averaged rate", round(av_rate,4))
            if av_rate >=fr - lim[n] and av_rate <fr + lim[n]:
                param_li.append(round(param,8))
                rate_li.append(av_rate)
                break
            elif av_rate < fr + lim[n]:
                param -= step
                mc+=1
            else:
                param += step
                pc+=1
            if mc>=2 and pc>=2:
                step = step*lr
                print("step", step)
                mc=0
                pc=0
            elif mc>5 or pc >5:
                step = step*(1+(1-lr))
                print("step", step)
                mc=0
                pc=0
    
    #     rate_li.append(av_rate)
#     param_li=[round(i,3) for i in param_li]
    print(param_li)
    print(rate_li)
    return param_li, rate_li

def make_spikes_w(peak,fx,fy,lx,ly,vrest,param_li,fr_name_li,  NE, Np, T, dt, offset, Tp, seed1):
    sw = "wake"
    dsb= "ISIM_{}_{}".format(seed1, dt)
    Lt = int((T +offset)/dt) 
    
    Ltp = int(Tp/dt)

    p = os.getcwd()
    savedir = p + "/mp_spikes/{}/{}/N_{}/".format(sw,dsb,NE)
    # savedir = p + "/net_search/mp_spikes/{}/{}/N_{}/".format(sw,dsb,NE)
    try:
        os.makedirs(savedir)
    except FileExistsError:
        print("{} is already exist".format(savedir))  

    for n, param in enumerate(param_li):
        fr_n = fr_name_li[n]

        spikes = gen_spikes_wake(NE, (T+offset), dt, param, seed1)
       # print("sp", spikes.shape)
        av_rate = round(np.mean(np.sum(spikes, axis = 1)/((T+offset)/1000)), 3)
        print("{} averaged rate: {}".format(fr_n, av_rate))

        post_spike = spikes[NE]
        spikes = spikes[0:NE]

        for iN in range(int(NE/Np)):
            mp_pres = np.zeros((Np, Lt))
            for n,spt in enumerate(spikes[iN*Np:(iN+1)*Np]):
                # print("n", n)
                mp=sp_to_mp(spt, Lt ,peak, fx,fy, lx,ly, vrest)
                mp_pres[n] = mp   
            for it in range(int(T/Tp)):
                mp_pre_f = savedir+"mp_pres_{}_{}_n{}t{}".format(sw, fr_n, iN, it)
                if it==0:
                    np.save(mp_pre_f, mp_pres[:,0:(it+1)*Ltp+int(offset/dt)])
                else:
                    np.save(mp_pre_f, mp_pres[:,(it)*Ltp+int(offset/dt):(it+1)*Ltp+int(offset/dt)])
                print("mp_pre_f", mp_pre_f)
    #
            del mp_pres
            gc.collect()
        del spikes
        gc.collect()

        mp_post=sp_to_mp(post_spike, Lt ,peak, fx,fy, lx,ly, vrest)
        for it in range(int(T/Tp)):
            mp_post_f = savedir+"mp_post_{}_{}_t{}".format(sw, fr_n, it)
            if it==0:
                np.save(mp_post_f, mp_post[0:(it+1)*Ltp+int(offset/dt)])
            else:
                np.save(mp_post_f, mp_post[(it)*Ltp+int(offset/dt):(it+1)*Ltp+int(offset/dt)])
            print("mp_post_f", mp_post_f)
        del mp_post
        del post_spike
        gc.collect()
    


if __name__ == "__main__":
    param_li, rate_li =spike_search_w(fr_li, fr_name_li, lim,  NE, T, dt, offset, lr, step, param, seed1)
    make_spikes_w(peak,fx,fy,lx,ly,vrest,param_li,fr_name_li,  NE, NE, T, dt, offset, Tp, seed1)

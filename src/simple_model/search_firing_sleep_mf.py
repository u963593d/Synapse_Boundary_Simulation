
import os
from scipy import interpolate
import gc
import numpy as np
import time
import shutil

#python3 search_firing_sleep_mf.py


#parameters for spike search
NE = 10 #number of neurons
T = 360000 #ms  1080000 for 0.03Hz, 2180000 for 0.01Hz
dt = 0.05  #step size
offset = 0  #ms

UP_th = 10 #minimum duration of burst
Tp=60000

furi = "ISIM"  #chaging parameter to adjust mean firing rates
ISI_M=1.5   #log10(mean of Inter-spike interval) ISIM
UP_M = 2.7 #log10(mean of Up state duration)  UPM
DOWN_M= 2.7  #log10(mean of Down state duration) DOWNM
seed1= 30

step=0.00002  #step size for chaging parameter 
lr = 0.97     #learning rate of step size for chaging parameter

fr_li =[10, 3, 1, 0.3, 0.1]  #list for mean firing rate
fr_name_li =["10.0","3.0","1.0", "0.30", "0.10"] #string list for mean firing rate
lim =[ 0.5, 0.5, 0.5, 0.05, 0.05] #precision for mean firing rates

# fr_li =[0.03]  #0.01
# fr_name_li =["0.03"]  #0.01
# lim =[0.05]


#spike parameters obtained from linear regression analysis of vivo data
a_IBI, b_IBI = 0.25,-0.35
a_dur, b_dur = 0.35,-0.7
a_ISI, b_ISI = -0.20, 0.95
a_updn, b_updn = -0.7, 4.0

DOWN_lim = -b_IBI /a_IBI
UP_lim = -b_dur/a_dur
ISI_lim = -b_ISI/a_ISI 

DOWN_lim2 = -(b_dur/a_dur)*a_updn + b_updn

print("DOWN_lim > ", DOWN_lim)
print("DOWN_lim2 < ", DOWN_lim2)

print("UP_lim > ", UP_lim)
print("ISI_lim <", ISI_lim)


#parameters for conversion from spike data to votage waveforms
peak =13                          #voltage peak of spikes (mv)
fx,lx =  int(-0.8/dt),  int(0.8/dt)  #duration of action potentials  (ms)
fy, lyup = -61, -61                  #membrane potential of Up state (mv)
fyd, ly = -76, -76                   #membrane potential of Down state (mv)
vrest = -70                          #resting potential  (mv)

# functions
def gen_spikes_sleep(N, T, dt, UP_th, DOWN_M, ISI_M, UP_M =None, seed = 15, txt_print=False):
    Lt = int(T/dt)
    np.random.seed(seed = seed)
    spikes=np.zeros((N+1, Lt),  dtype = "int8")
    spikes_ID=np.zeros((N+1, Lt), dtype = "int8")

    bt_li=[] #time of burst
    UP_li=[]

    total=0
    DOWN_SD = a_IBI*DOWN_M + b_IBI 
    ISI_SD  =a_ISI*ISI_M + b_ISI 
    
    if UP_M == None: 
        UP_M = (DOWN_M-b_updn)/a_updn
        UP_SD = a_dur*UP_M + b_dur 
        
    else:
        UP_SD = a_dur*UP_M + b_dur
        
#     print("UP_SD", UP_SD)

    while(1):
        IBI = 10**(np.random.normal(DOWN_M,DOWN_SD,1)[0])/dt #ms
        total += IBI
        if total > Lt:#Lt
            break

        burst_s = total
        while(1):
            dur = 10**(np.random.normal(UP_M,UP_SD,1)[0])/dt #ms
            if dur > UP_th:
                total += dur
                break  # dur limit

    #     print("IBt", IBt)
        if total > Lt:#Lt
            break
            bt_li.append([burst_s, Lt-1])
            UP_li.append(Lt-1-burst_s)
        else:
            bt_li.append([burst_s, total])
            UP_li.append(dur)
    #print("bt_li", bt_li)

    burst_n=len(bt_li)     
    burst_ID = np.arange(0,burst_n).astype("int32").tolist()
    mean_dur = np.mean(UP_li)
    if txt_print ==True:
        print("mean dur (ms)", mean_dur*dt)
        print("total bursts", burst_n)
        print("burst frequency", burst_n/((T)/1000))

    maxfre_n_li = np.zeros(N+1)
    
    total_dur = np.sum(UP_li)*dt
    print("total_dur", total_dur)
    for n in range(N+1):
        total_sp = 0
        for j in range(burst_n):
            #print("j",j)
            bti=bt_li[j]
            duri=UP_li[j]
            #print("duri", duri)
            ISt=0
            ISt_pre = 0
            c=0
            while(1):
              #  if n!=N:
                ISt +=  10**(np.random.normal(ISI_M,ISI_SD,1)[0])/dt #ms
               # print("ISI", ISt)
                if ISt <= duri:
    #                 if bti[0]+ISt < Lt:
                    spikes[n][int(bti[0]+ISt)]=1
                    ISt_pre = ISt
                    spikes_ID[n][int(bti[0]+ISt)] = c  #spike ID in burst
                    c+=1
                       # print("sp")
    #                 elif bti[0]+ISt >Lt and c==0:
    #                     ISt=0
                else:
                    if c>1: # and ISt > duri:#ms
                        spikes_ID[n][int(bti[0]+ISt_pre)] = -1 
                    elif c==1: # and ISt > duri:#ms
                        spikes_ID[n][int(bti[0]+ISt_pre)] = -2
                    total_sp += c
                    break
        
#         if n != N:
#             print("total_sp",total_sp)
        maxfre_n_li[n] = total_sp/(total_dur/1000)  #Hz
                
    maxfre = np.mean(maxfre_n_li)
    print("maxfre", maxfre)
    post_spike = spikes[N]
    spikes = spikes[0:N]
    post_ID = spikes_ID[N]
    spikes_ID = spikes_ID[0:N]
    return spikes, post_spike, spikes_ID, post_ID, maxfre


#convert spike trains to voltage waveform
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


def spike_search(peak, fx, fy,fyd, lx, ly, lyup, vrest, fr_li, fr_name_li, lim, furi, NE, T, dt, offset, lr, step, ISI_M, DOWN_M, UP_M, seed1):
    Lt = int((T +offset)/dt) #0.1ms
    UP_th  = 10
    lim = [i/10 for i in lim]
    tlim=90  #min
    
    param_all_li = {} 
    rate_all_li = {} 

    pc=0
    mc=0
    
    if furi =="ISIM":
        param = ISI_M
    elif furi=="DOWNM":
        param = DOWN_M
    elif furi=="UPM":
        param = UP_M
        
#     a_IBI, b_IBI = 0.3, -0.50 
#     a_dur, b_dur = 0.3, - 0.5
#     a_ISI, b_ISI = -0.25, 1.0
#     a_updn, b_updn = -0.60, 4.0
    a_IBI, b_IBI = 0.25,-0.35#0.3, -0.50 
    a_dur, b_dur = 0.35,-0.7#0.3, - 0.5
    a_ISI, b_ISI = -0.20, 0.95
    a_updn, b_updn = -0.7, 4.0#-0.60, 4.0

    DOWN_lim = -b_IBI /a_IBI
    UP_lim = -b_dur/a_dur
    ISI_lim = -b_ISI/a_ISI
#     end_f = 0
    seed_N=100
    for N in range(seed_N):
        rate_li = {}
        param_li= {}
        fr_li2 = []
        fr_name_li2= []
        if len(list(param_all_li.keys()))==5:
            break
        
        seed1=np.random.randint(10**3)
        print("seed", seed1)
        
        st= time.time()
        tout = 0
        if furi =="ISIM" or furi =="DOWNM":
            for n,fr in enumerate(fr_li):
                if fr_name_li[n] in param_all_li:
                    continue
                print("search:{} Hz".format(fr))

                if tout==1:
                    break
                    
                
                count=0
                while(1):
                    count+=1
                    et=time.time()
                    if et-st > tlim *60: #sec
                        tout = 1
                        print("{} Hz time out".format(fr))
                        break
#                     print("param", param)
        
                    if furi =="ISIM":
                        if param > ISI_lim:
                            break
                        spikes, post_spike,spikes_ID, post_ID, maxfre = gen_spikes_sleep(NE, (T+offset), dt, UP_th, DOWN_M, param, UP_M, seed1)

                    elif furi =="DOWNM":
                        if param <= DOWN_lim:
                            break
                        spikes, post_spike,spikes_ID, post_ID, maxfre = gen_spikes_sleep(NE, (T+offset), dt, UP_th, DOWN_M, param, UP_M, seed1)

                    else:
                        if param <=  UP_lim:
                            break
                        spikes, post_spike, spikes_ID, post_ID, maxfre = gen_spikes_sleep(NE, (T+offset), dt, UP_th, param, ISI_M, UP_M, seed1)
                    print("maxfre", maxfre)
                    av_rate = round(np.mean(np.sum(spikes, axis = 1)/((T+offset)/1000)), 4) #averaged rate
                    if count <3:
                        print("averaged rate", av_rate)
                        print("maxfre", maxfre)
                    if maxfre >=fr - lim[n] and maxfre <fr + lim[n]:
#                         param_li.append(round(param,8))
                        param_li[fr_name_li[n]] = round(param,10)
                        param_all_li[fr_name_li[n]] = round(param,10)
#                         rate_li.append(av_rate)
                        rate_li[fr_name_li[n]] = maxfre
                        rate_all_li[fr_name_li[n]] = maxfre
                        fr_name_li2.append(fr_name_li[n])
                        fr_li2.append(fr_li[n])
                        print("averaged rate", av_rate)
                        print("maxfre", maxfre)
                        break
                    elif maxfre < fr - lim[n]:
                        param -= step
                        mc+=1
                    else:
                        param += step
                        pc+=1
                    if mc>=4 and pc>=4:
                        step = step*lr
#                         print("step", step)
                        mc=0
                        pc=0
                    elif mc>5 or pc >5:
                        step = step*(1+(1-lr))
#                         print("step", step)
                        mc=0
                        pc=0
        

        else:
            for n,fr in enumerate(fr_li):
                if fr_name_li[n] in param_li:
                    continue
                print("search:{} Hz".format(fr))
                count=0
                while(1):
                    count+=1
#                     print("param", param)
                    if param <=  UP_lim:
                        break# UP_lim = -b_dur/a_dur
                    spikes, post_spike,spikes_ID, post_ID, maxfre = gen_spikes_sleep(NE, (T+offset), dt, UP_th, DOWN_M, ISI_M, param, seed1)
                    av_rate = round(np.mean(np.sum(spikes, axis = 1)/((T+offset)/1000)), 4) #averaged rate
                    print("maxfre", maxfre)
                    if count <3:
                        print("averaged rate", av_rate)
                        print("maxfre", maxfre)
                    if maxfre >=fr - lim[n] and maxfre <fr + lim[n]:
                        param_li[fr_name_li[n]] = round(param,10)
                        param_all_li[fr_name_li[n]] = round(param,10)
#                         rate_li.append(av_rate)
                        rate_li[fr_name_li[n]] = maxfre
                        rate_all_li[fr_name_li[n]] = maxfre
                        fr_name_li2.append(fr_name_li[n])
                        fr_li2.append(fr_li[n])
                        print("averaged rate", av_rate)
                        print("maxfre", maxfre)
                        break
                    elif maxfre < fr - lim[n]:
                        param -= step
                        mc+=1
                    else:
                        param += step
                        pc+=1
                    if mc>=4 and pc>=4:
                        step = step*lr
#                         print("step", step)
                        mc=0
                        pc=0
                    elif mc>5 or pc >5:
                        step = step*(1+(1-lr))
#                         print("step", step)
                        mc=0
        
        param_v = list(param_li.values())
        print("param_v", param_v)
        if len(param_v)!=0:
            if furi == "ISIM":
                mp_spikes_sleep(param_v, peak, fx, fy,fyd, lx, ly, lyup, vrest,fr_li2, fr_name_li2, "ISIM", NE, NE, T, dt, offset, Tp, DOWN_M, UP_M, seed1)
            elif furi == "DOWNM":
                mp_spikes_sleep(param_v, peak, fx, fy,fyd, lx, ly, lyup, vrest,fr_li2, fr_name_li2, "DOWNM", NE, NE, T, dt, offset, Tp, ISI_M, UP_M, seed1)
            else:
                mp_spikes_sleep(param_v, peak, fx, fy,fyd, lx, ly, lyup, vrest,fr_li2, fr_name_li2, "UPM", NE, NE, T, dt, offset, Tp, ISI_M, DOWN_M, seed1)
    
    
    #integrate folders
    mp_dir = "mp_spikes_mf/sleep/"
    try:
        os.makedirs(mp_dir)
    except FileExistsError:
        print("{} is already exist".format(mp_dir))  

    fols = os.listdir(mp_dir)
    
    if furi =="ISIM":
        dsb= "DOWNM{}_UPM{}".format(DOWN_M, UP_M)
    elif furi =="DOWNM":
        dsb= "ISIM{}_UPM{}".format(ISI_M, UP_M)
    elif furi =="UPM":
        dsb= "ISIM{}_DOWNM{}".format(ISI_M, DOWN_M)
    
    fol_dsbs = []
    for fol in fols:
        if dsb in fol:
            fol_dsbs.append(fol)
    print("fol_dsbs", fol_dsbs)
    root_fol =mp_dir + "{}/N_{}/".format(fol_dsbs[0], NE)
    
    for n, fol_dsb in enumerate(fol_dsbs):
        if n==0:
            continue
        pre_fol = mp_dir + "{}/N_{}".format(fol_dsb, NE)
        files = os.listdir(pre_fol+"/")
        for f in files:
            shutil.copy(pre_fol +"/"+ f, root_fol + f)
        shutil.rmtree(pre_fol)
    
    print(param_all_li)
    print(rate_li)
    return param_all_li, rate_li

def mp_spikes_sleep(param_li, peak, fx, fy,fyd, lx, ly, lyup, vrest, fr_li, fr_name_li, furi, NE, Np, T, dt, offset, Tp, cons1, cons2, seed1):
    Lt = int((T +offset)/dt) #0.1ms
    Ltp = int(Tp/dt)
    sw = "sleep"
    
    if furi =="ISIM":
        dsb= "DOWNM{}_UPM{}_{}_{}".format(cons1, cons2, seed1, dt)
    elif furi =="DOWNM":
        dsb= "ISIM{}_UPM{}_{}_{}".format(cons1, cons2, seed1, dt)
    elif furi =="UPM":
        dsb= "ISIM{}_DOWNM{}_{}_{}".format(cons1, cons2, seed1, dt)


    p = os.getcwd()
#     savedir = p + "/mp_spikes/{}/diff10_40_0.05_2/N_{}/".format(sw,NE)
    savedir = p + "/mp_spikes_mf/{}/{}/N_{}/".format(sw,dsb,NE)

    try:
        os.makedirs(savedir)
    except FileExistsError:
        print("{} is already exist".format(savedir))  

    rate_li=[]

    for n, param in enumerate(param_li):
        fr_n = fr_name_li[n]
        
        #chaging furi to adjust spike firing rate
        if furi =="ISIM":
            spikes, post_spike,spikes_ID, post_ID, dur = gen_spikes_sleep(NE, (T+offset), dt, UP_th, cons1, param, cons2, seed1)

        elif furi =="DOWNM":
            spikes, post_spike,spikes_ID, post_ID, dur = gen_spikes_sleep(NE, (T+offset), dt, UP_th, param, cons1, cons2, seed1)

        elif furi =="UPM":
            spikes, post_spike,spikes_ID, post_ID, dur = gen_spikes_sleep(NE, (T+offset), dt, UP_th, cons2, cons1, param, seed1)

       # print("sp", spikes.shape)
        av_rate = round(np.mean(np.sum(spikes, axis = 1)/((T+offset)/1000)), 3)
        print("{} averaged rate: {}".format(fr_n, av_rate))
        rate_li.append(av_rate)

        for iN in range(int(NE/Np)):
            mp_pres = np.zeros((Np, Lt))
            for n,spt in enumerate(spikes):
                spt_ID = spikes_ID[n]
                mp=sp_to_mp(spt, spt_ID, Lt ,peak, fx, fy,fyd, lx, ly, lyup, vrest)
                mp_pres[n] = mp

            for it in range(int(T/Tp)):
                mp_pre_f = savedir+"mp_pres_{}_{}_n{}t{}".format(sw, fr_n, iN, it)
                if it==0:
                    np.save(mp_pre_f, mp_pres[:,0:(it+1)*Ltp+int(offset/dt)])
                else:
                    np.save(mp_pre_f, mp_pres[:,(it)*Ltp+int(offset/dt):(it+1)*Ltp+int(offset/dt)])
                print("mp_pre_f", mp_pre_f)
   
            del mp_pres
            gc.collect()
        del spikes
        gc.collect()

        mp_post=sp_to_mp(spt, spt_ID, Lt ,peak, fx, fy,fyd, lx, ly, lyup, vrest)
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
    print(param_li)
    print(rate_li)
    


if __name__ == "__main__":
    param_li, rate_li = spike_search(peak, fx, fy,fyd, lx, ly, lyup, vrest, fr_li, fr_name_li, lim, furi, NE, T, dt, offset, lr, step, ISI_M, DOWN_M, UP_M, seed1)

# -*- coding: utf-8 -*-



import subprocess
import os
import gc 
import sys
# """
# LIMIT THE NUMBER OF THREADS!
# change local env variables BEFORE importing numpy
# """
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

from enum import Flag, auto
import numpy as np
from scipy.signal import periodogram, find_peaks
from scipy import signal
from scipy.stats import pearsonr
from typing import Optional
import matplotlib.pyplot as plt
from multiprocessing import Pool, Array, Manager
import traceback
# import models
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fs=25
fs_l=18

class CV_S:
    def __init__(self, Sini, LA,UA, tau_i, tau_d,  T, offset, dt):
        self.T = T
        self.offset = offset
        self.dt = dt
        self.LA = LA
        self.UA = UA
        self.tau_i = tau_i
        self.tau_d = tau_d
        self.init = np.array([Sini])
        
#         self.sol_num = solvers

    def dAlldt1(self, init):
        s = init[0]
        dsdt = -(s-self.LA)/self.tau_d 
        ode = np.array([dsdt])
        return ode
    
    def dAlldt0(self, init):
        s = init[0]
        dsdt = (self.UA-s)/self.tau_i 
        ode = np.array([dsdt])
        return ode
    
    def Solvers(self, func, init, solver=0):
        # 4th order Runge-Kutta 法
        
        if solver == 1:
            k1 = self.dt*func(init)
            k2 = self.dt*func(init + 0.5*k1)
            k3 = self.dt*func(init + 0.5*k2)
            k4 = self.dt*func(init + k3)
            return init + (k1 + 2*k2 + 2*k3 + k4) / 6
            # 陽的 Euler 法
        elif solver == 0:
            return init + self.dt*func(init)
        else:
            return None
    
    def diff_S(self, cv_sw):
        Lt = len(cv_sw)


        X_arr = np.zeros((1, Lt))
        X_arr[:,0] = self.init

        for it in range(Lt): 
            if it==0:
                continue
            
            if cv_sw[it]==0:
                init = self.Solvers(self.dAlldt0, self.init, 1)
            elif cv_sw[it]==1:
                init = self.Solvers(self.dAlldt1, self.init, 1)
                
            self.init = init
#             print("init", init)
#             if cv_sw[it]==0:
#                 X_arr[:,it] = init#self.UA - init
#             elif cv_sw[it]==1:
            X_arr[:,it] = init #init + self.LA 
        return X_arr[0,:]

class WavePattern(Flag):
    """ Enumeration class that distinguish different wave pattern.
    """
    SWS = auto()
    SWS_FEW_SPIKES = auto()
    SWS_HIGH_FR = auto()
    AWAKE_HIGH_FR = auto()
    AWAKE = auto()
    RESTING = auto()
    EXCLUDED = auto()
    ERROR = auto()
    cyclic_firing_with_weak_synaptic_currents = auto()

class WaveCheck:
    """ Check which wave pattern the neuronal firing belong to.

    Parameters
    ----------
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)
    
    Attributes
    ----------
    wave_patters : WavePattern
        choices of wave pattern: enumeration objects
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)
    freq_spike : FreqSpike
        contains helper functions for analyzing firing pattern
        using those frequency and spikes
    """
    def __init__(self, samp_freq, ori_ampai_max: Optional[float]=None, ampai_max: Optional[float]=None, ori_v_min: Optional[float]=None) -> None:
        self.wave_pattern = WavePattern
        self.samp_freq = samp_freq
        self.freq_spike = FreqSpike(samp_freq=samp_freq)
        #self.down = Down_Judge_Null(100000, 10, 99999, 999999)
        
        self.ori_ampai_max = ori_ampai_max
        self.ampai_max = ampai_max
        self.ori_v_min = ori_v_min
        
    
    def pattern(self, v: np.ndarray, T, dt, pr=False) -> WavePattern:
        """

        Parameters
        ----------
        v : np.ndarray
            membrane potential over time

        Returns
        ----------
        WavePattern
            which wave pattern `v` belong to 
        """
        if np.any(np.isinf(v)) or np.any(np.isnan(v)):
            return self.wave_pattern.EXCLUDED
        detv: np.ndarray = signal.detrend(v)
        max_potential: float = max(detv)
        f: np.ndarray  # Array of sample frequencies
        spw: np.ndarray  # Array of power spectral density or power spectrum
        f, spw = periodogram(detv, fs=1/dt*1000)
        maxamp: float = max(spw)
        nummax: int = spw.tolist().index(maxamp)
        maxfre: float = f[nummax]
#         plt.plot(v)
#         plt.show()
     
        numfire: float = self.freq_spike.count_spike(v)
        if pr ==True:
            print("maxfre", maxfre)
            print("numfire", numfire)
        
        ot = float(T/1000)  #sec
        
        if np.min(v) < -200 or  np.max(v) >200:
            return self.wave_pattern.EXCLUDED
        elif (maxfre < 0.5) or (numfire < 0.6*ot) or np.max(v)<-30 or np.min(v)>-30:
            return self.wave_pattern.RESTING
        elif (0.5 < maxfre) and (numfire > ot*2*maxfre):
            if numfire < 30*ot:
                return self.wave_pattern.SWS
            elif numfire >= 30*ot  and numfire < 100*ot:
                return self.wave_pattern.SWS_HIGH_FR
            else:
                return self.wave_pattern.EXCLUDED
        
        elif (0.5 < maxfre) and numfire <= ot * 2*maxfre:  # and numfire > ot * 0.5*maxfre  # (np.sum(v < -70)/len(v))*100 < 0.1:
            
                if numfire < 30*ot:
                    return self.wave_pattern.AWAKE
                elif numfire >= 30*ot and numfire < 100*ot:
                    return self.wave_pattern.AWAKE_HIGH_FR
                else:
                    return self.wave_pattern.EXCLUDED
        else:
            return self.wave_pattern.EXCLUDED


class FreqSpike:
    """ 

    Parameters
    ----------
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)

    Attributes
    ----------
    samp_freq : int
        sampling frequency of neuronal recordings (Hz)
    """
    def __init__(self, samp_freq: int) -> None:
        self.samp_freq = samp_freq

    def count_spike(self, v: np.ndarray) -> int:
        """ Count how many times a neuron fired.

        If neuron traverse -20 mV in a very short time range (1ms), 
        traverse count is added 1. Here, spike count is calculated as 
        traverse count // 2. 

        Parameter
        ---------
        v : np.ndarray
            membrane potential of a neuron
        
        Return
        ---------
        int
            spike count
        """
        ntraverse: int = 0
        ms: int = int(self.samp_freq / 1000)
        for i in range(len(v)-1):
            if (v[i]+20) * (v[i+ms]+20) < 0:
                ntraverse += 1
        nspike: int = int(ntraverse//2)
#         print("nspike", nspike) 

#         peaks = find_peaks(v, height=-20)[0]
#         nspike = len(peaks)
        return nspike


fs=25
fs_l=18


class eval_wave:
    def __init__(self,model_name, date_i, NE, NI,T, Tp, dt,drc, fig_dir, savef, ex, ex_in,move, w, LA, UA, ti, td):
        manager = Manager()
        self.LA=LA
        self.UA=UA
        self.ti=ti
        self.td=td
        self.model = model_name
        self.date_i = date_i
        self.N = NE + NI
        self.NE = NE
        self.NI = NI
        self.ex = ex
        self.ex_in = ex_in
        self.tbs = int(T/Tp)
     
        self.T = T
        self.Tp=Tp
        self.dt = dt
        self.move=move 
        self.w = w

        self.drc = drc
        self.drc_neu = drc+"neu/"
        self.drc_syn = drc+"syn/"

        try:
            v_file = self.drc_neu+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(0)+"_"+str(0)+".bin" 
            f2= open(v_file, "rb")
            rectype = np.dtype(np.float64)
            v_tb = np.fromfile(f2, dtype=rectype)
            self.dt = round(self.Tp/v_tb.shape[0],4)
        except:
            print("file_Error, ex{}".format(ex))
            self.dt = dt
            traceback.print_exc()
            
        print("self.dt", self.dt)
        
        self.x = np.arange(0, T, self.dt)/1000
        # self.labels = ["S00", "S11", "S01", "S10"]
        
        self.counts = int((T-w)/move +1)
        
        
        # self.labels = ["S00", "S11", "S01", "S10"]
        # counts = int((T-w)/move +1)
        self.cv_li = manager.Array('d', range(self.counts))
        self.fre_li = manager.Array('d', range(self.counts))  
        
#         self.df_np = np.zeros((len(exs),13))
        # self.df_np = manager.Array('d', range(len(exs)*14)
        
            
        self.fig_dir = fig_dir
        self.savef =savef
        # else:
            # self.savef = fig_dir+"/{}/{}_NE{}_NI{}_conth{}_{}_{}_{}_gnap{}_gabac{}_T{}_dt{}".format(model, self.date_i, self.NE,self.NI,self.con_ex_ex,self.con_ex_in,self.con_in_ex, self.con_in_in,self.g_nap_SD,self.gabac,self.T,self.dt)
    
    
    def calc_cv(self, N, spts, T, w_cv,  move_cv):
        wc = int((T-w_cv)/move_cv+1)
        sp_li = []
        for m in range(wc):
            if m*move_cv/self.dt>0 and m*move_cv/self.dt + w_cv/self.dt  < (T)/self.dt:
                sp_sum = np.sum(spts[0:N,int(m*move_cv/self.dt):int(m*move_cv/self.dt + w_cv/self.dt)])
                sp_li.append(sp_sum)
        if np.mean(sp_li)!=0:
            sps_cv = np.std(sp_li)/np.mean(sp_li)
        else:
            sps_cv =0
    #     print("CV_sp_count/w", sps_cv)
        return sps_cv

    def raster_plot(self, drc_neu, N, ex,ex_in,T, off):
        pdf = PdfPages(self.savef+"_raster.pdf")
        spts = np.zeros((N, int((T-off)/self.dt)), dtype = "int8")
        for n in range(N):
            # try:
                for tb in range(self.tbs):
                    v_file = drc_neu+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
                    f2= open(v_file, "rb")
                    rectype = np.dtype(np.float64)
                    v_tb = np.fromfile(f2, dtype=rectype)
                    if tb==0:
                        v=v_tb
                    else:
                        v=np.append(v, v_tb)
                    f2.close()
                pattern, maxfre, sp_c, peak_np = self.fre_spike(v[int(off/self.dt):int(T/self.dt)], int(T-off), self.dt)
                spts[n]= peak_np
        fig, axes = plt.subplots(figsize=(20, 8))
        Lt = int(T/self.dt)
        x2=  np.arange(0,Lt,1)
#                 spT = spts[spts.sum(axis=1) > 0., :]  #spikeあるものだけ選ぶ
        #color_set = ['r', 'b', 'k', 'orange', 'c']
        for i in range(spts.shape[0]):
            t_sp = x2[spts[i, :] > 0.5]   # spike times
            axes.plot(t_sp*self.dt/1000, i*np.ones(len(t_sp)), '.',
                    ms=3, markeredgewidth=0.1, color ="black")
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('Neuron ID', fontsize = fs)
        #plt.xlim(0, 1)
        axes.tick_params(labelsize=fs_l)

        # axes.set_ylim(-110,40)
        axes.set_rasterized(True)
        pdf.savefig(fig)
        plt.close()
        pdf.close()
        
    
    def multi_cal(self, cores, drc_neu, N, ex,ex_in, T, move, w, move_cv, w_cv):
        args = []
        # self.move=move
        # self.w = w
        # counts = int((T-w)/move +1)
        # self.cv_li = np.zeros(counts)
        # self.counts=counts
        cp=80 
        blockdim_x=8 
        blockdim_y=8
        outc= "/mnt/gpu_data/net_search/CV_gf {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(drc_neu, self.savef, ex, ex_in, N, T, self.Tp, self.dt, move, w, move_cv, w_cv, cp, blockdim_x, blockdim_y)
        subprocess.run([outc], shell=True)
        
    
    def cv_th_data(self, n_conv, cv_th):
        pdf = PdfPages(self.savef+"_cvplot.pdf")
        b = np.ones(n_conv)/n_conv 
        # print("cv_li_d", len(self.cv_li_d
        
        f2= open(self.savef + "_cv_ori.bin", "rb")
        rectype = np.dtype(np.float64)
        self.cv_li  = np.fromfile(f2, dtype=rectype)
        f2.close()
        # np.save(self.savef+"_cv_ori", np.array(self.cv_li))
        print("cv_ori_li", self.cv_li)
        # np.save(self.savef+"_spfre", np.array(self.fre_li))
        
        cv_ave=np.convolve(self.cv_li, b, mode="same") 
        
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(cv_ave)   #time[0:int(T/dt)-1],
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('CV_ave',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        # axes.set_ylim(-110,40)
        axes.set_rasterized(True)
        pdf.savefig(fig)
        pdf.close()
        plt.close()
        # dcv=np.max(cv_ave)-np.min(cv_ave)
        # f=open(self.savef+"_d_cv.txt", "w")
        # f.write(str(dcv))
        # f.close()
    
        sw_cv=cv_ave>cv_th
        
        # cv_m=np.mean(self.cv_li)
        # cv_std=np.std(self.cv_li)
        # cv_norm=(self.cv_li-cv_m)/cv_std
        
        
        return sw_cv

    def sw_time(self,move,sw_cv, cv_th):
        sleep_T_li = []
        wake_T_li = []
        sleep_m_li =[]
        wake_m_li =[]
        sleep_v_li =[]
        wake_v_li =[]
        sw_ratio_li =[]
        
        shifted_arr = np.concatenate(([sw_cv[0]], sw_cv[:-1]))

        # 連続した0または1の長さを保持するリストを初期化する
        lengths = []
        sw_la=[]

        # 現在の連続した数と、前回の値を初期化する
        current_length = 1
        previous_value = sw_cv[0]
        if previous_value==1:
            sw_la.append(1)
        else:
            sw_la.append(0)

        # 配列をループして、連続した値の数を計算する
        for value, shifted_value in zip(sw_cv, shifted_arr):
            if value == shifted_value:
                # 値が前回と同じ場合は、連続した数を1増やす
                current_length += 1
            else:
                # 値が前回と異なる場合は、現在の連続した数をリストに追加し、新しい数の計算を開始する
                lengths.append(current_length)
                if value==1:
                    sw_la.append(1)
                else:
                    sw_la.append(0)

                current_length = 1
                previous_value = value

        # 最後に、最後の連続した数をリストに追加する
        lengths.append(current_length)

        # 連続した数のリストを表示する
        print("lengths", lengths)
        cumsum  = [0]
        psum = 0
        for n, l  in enumerate(lengths):
            if n==len(lengths)-1:
                continue
            psum += l
            cumsum.append(psum)
        
       
       #length -> 
        durs =[]
        for n, l  in enumerate(lengths):
            s_n = int(cumsum[n]*move/self.dt)
            end_n = s_n + int(lengths[n]*move/self.dt)
            durs.append([s_n, end_n])
        
        sleep_durs = [durs[n] for n, sw in enumerate(sw_la) if sw==1 and lengths[n]>=10]
        wake_durs = [durs[n] for n, sw in enumerate(sw_la) if sw==0 and lengths[n]>=10]
        print("sleep_period", sleep_durs)
        print("wake_period", wake_durs)
        
        len_T = [i*move/1000 for i in lengths]  #s 
        sleep_time=[len_T[n] for n, sw in enumerate(sw_la) if sw==1 and lengths[n]>=10 and n!=0 and n!=len(lengths)-1]
        wake_time=[len_T[n] for n, sw in enumerate(sw_la) if sw==0 and lengths[n]>=10 and n!=0 and n!=len(lengths)-1]

        cv_area_li= np.zeros(int(self.T/self.dt)) #np.empty(0)
        sumT=self.w/2/self.dt#0
        for i, L in enumerate(len_T):
            if sw_la[i]==0:
                cv_area = np.zeros(int(L*1000/self.dt))
            elif sw_la[i]==1:
                cv_area = np.ones(int(L*1000/self.dt))
            if int(sumT+L*1000/self.dt) >len(cv_area_li):
                last=len(cv_area_li)
                cv_area_li[int(sumT):last] = cv_area[:last-int(sumT)]
            else:
                last=int(sumT+L*1000/self.dt)
                cv_area_li[int(sumT):last] = cv_area
            sumT += int(L*1000/self.dt)
        
        print("cv_area_li", len(cv_area_li))
        np.save(self.savef + "_cvsw", cv_area_li)
        
        # LA=0
        # UA=1.0
        # tau_i=10000
        # tau_d=10000
        T2 = self.T-self.w  #ms
        offset = int(self.w/2)  #ms
        cv_area_li = cv_area_li[int(offset/self.dt):int((self.T-offset)/self.dt)]
        if not os.path.exists(self.savef + "_pro_s_{}_{}_{}_{}.npy".format(self.LA,self.UA, self.ti, self.td)):
            if cv_area_li[0]==0:
                Sini = 0#self.UA - init
            elif cv_area_li[0]==1:
                Sini = 1

            pro_s = CV_S(Sini, self.LA,self.UA, self.ti, self.td,  T2, offset, self.dt)
            X_arr = pro_s.diff_S(cv_area_li) 
            
            np.save(self.savef + "_cvsw", cv_area_li)
            np.save(self.savef + "_pro_s_{}_{}_{}_{}".format(self.LA,self.UA, self.ti, self.td), X_arr)
            pdf = PdfPages(self.savef+"_cvsw_{}_{}_{}_{}.pdf".format(self.LA,self.UA, self.ti, self.td))
            fig, axes = plt.subplots(figsize=(20, 8))
            axes.plot(cv_area_li)   #time[0:int(T/dt)-1],
            axes.set_xlabel('Time (sec)',fontsize = fs)
            axes.set_ylabel('CV_sw',fontsize = fs)
            axes.tick_params(labelsize=fs_l)
            # axes.set_ylim(-110,40)
            axes.set_rasterized(True)
            pdf.savefig(fig)
            
            fig, axes = plt.subplots(figsize=(20, 8))
            axes.plot(X_arr)   #time[0:int(T/dt)-1],
            axes.set_xlabel('Time (sec)',fontsize = fs)
            axes.set_ylabel('Process S',fontsize = fs)
            axes.tick_params(labelsize=fs_l)
            axes.set_ylim(0, 1.1)
            axes.set_rasterized(True)
            pdf.savefig(fig)
            pdf.close()
            plt.close()
        else:
            X_arr = np.load(self.savef + "_pro_s_{}_{}_{}_{}.npy".format(self.LA,self.UA, self.ti, self.td))
        
        print("len(X_arr)", len(X_arr))
        
        if sw_la[0]==0:
            num=len(wake_time)
        else:
            num=len(sleep_time)
            
        sleep_T=np.sum(sleep_time[0:num])
        wake_T=np.sum(wake_time[0:num])
        sleep_m=np.mean(sleep_time)
        wake_m=np.mean(wake_time)
        sleep_v=np.var(sleep_time)
        wake_v=np.var(wake_time)
        sw_ratio = sleep_T/wake_T

        print("total_sleep (sec)", sleep_T)
        print("total_wake (sec)", wake_T)
        print("mean_sleep (sec)", sleep_m)
        print("meanl_wake (sec)", wake_m)
        print("var_sleep", sleep_v)
        print("var_wake", wake_v)
        print("s/w ratio", sleep_T/wake_T)
        
        sleep_T_li.append(sleep_T)
        wake_T_li.append(wake_T)
        sleep_m_li.append(sleep_m)
        wake_m_li.append(wake_m)
        sleep_v_li.append(sleep_v)
        wake_v_li.append(wake_v)
        sw_ratio_li.append(sw_ratio)
        
        df_sw = pd.DataFrame(list(zip(sleep_T_li, wake_T_li,sleep_m_li, wake_m_li,sleep_v_li, wake_v_li, sw_ratio_li)), columns=["total_sleep", "total_wake", "mean_sleep", "mean_wake", "var_sleep", "var_wake","sw_ratio"])
        df_sw.to_excel(self.savef + "_cvth{}_cc.xlsx".format(cv_th))
         
        
        c=0
        for post_i in range(self.N):
            if c==1:
                break
            for s in range(self.N):
                cpre_file = self.drc_syn+  "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"cpre"+str(s)+"-"+str(post_i)+"_"+str(0)+".bin"
                if os.path.exists(cpre_file):
    #                 print(cpre_file)
                
                    for tb in range(self.tbs):
                        sp_file = self.drc_syn+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"Sp"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
                        fsp= open(sp_file, "rb")
                        rectype = np.dtype(np.float64)
                        sp_tb = np.fromfile(fsp, dtype=rectype).astype("float32")
                        if tb==0:
                            sp=sp_tb
                        else:
                            sp=np.append(sp, sp_tb).astype("float32")
                        fsp.close() 
                        
                    for tb in range(self.tbs):
                        k_file = self.drc_syn+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"K"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
                        fk= open(k_file, "rb")
                        rectype = np.dtype(np.float64)
                        k_tb = np.fromfile(fk, dtype=rectype).astype("float32")
                        if tb==0:
                            k=k_tb
                        else:
                            k=np.append(k, k_tb).astype("float32")
                        fk.close()
                    
                    c+=1
                    if c==1:
                        break
        #mean r
        # sum_sp=np.zeros(int(self.T/self.dt))
        # c=0
        # for post_i in range(self.N):
        #     for s in range(self.N):
        #         cpre_file = self.drc_syn+  "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"cpre"+str(s)+"-"+str(post_i)+"_"+str(0)+".bin"
        #         if os.path.exists(cpre_file):
                        
        #             for tb in range(self.tbs):
        #                 sp_file = self.drc_syn+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"Sp"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
        #                 fsp= open(sp_file, "rb")
        #                 rectype = np.dtype(np.float64)
        #                 sp_tb = np.fromfile(fsp, dtype=rectype).astype("float32")
        #                 if tb==0:
        #                     sp=sp_tb
        #                 else:
        #                     sp=np.append(sp, sp_tb).astype("float32")
        #                 fsp.close()
            
                    
        #             sum_sp=sum_sp + sp
        #             c+=1
        # mean_sp=sum_sp/c
                    
        # corr, pv = pearsonr(cv_area_li[int(self.w/2/self.dt):int(len(cv_area_li)-self.w/2/self.dt)], sp[int(self.w/2/self.dt):int(len(cv_area_li)-self.w/2/self.dt)])
        # corr_a, pv_a = pearsonr(cv_area_li[int(self.w/2/self.dt):int(len(cv_area_li)-self.w/2/self.dt)], k[int(self.w/2/self.dt):int(len(cv_area_li)-self.w/2/self.dt)])
        
        corr, pv = pearsonr(X_arr, sp[int(offset/self.dt):int((self.T-offset)/self.dt)])
        corr_a, pv_a = pearsonr(X_arr, k[int(offset/self.dt):int((self.T-offset)/self.dt)])
        
        
        print("corr", corr)
        print("pv", pv)
        print("corr_a", corr_a)
        print("pv_a", pv_a)
        np.savetxt(self.savef+"_corr_pros_{}_{}_{}_{}.csv".format(self.LA,self.UA, self.ti, self.td), np.array([corr, pv, corr_a, pv_a]), delimiter=',')
        

        return df_sw, sleep_durs, wake_durs
    
    def weights_cal(self, Gex, stim_start, stim_dur):
        pdf = PdfPages(self.savef+"_weights.pdf")
        syN=0
        pairs = []
        for post_i in range(self.NE+self.NI):
            for s in range(self.NE+self.NI):
                cpre_file = self.drc_syn+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+ "cpre"+str(s)+"-"+str(post_i)+"_"+str(0)+".bin"
                if os.path.exists(cpre_file):
                    pairs.append([s,post_i])
                    syN += 1
                    
        con_mat = np.zeros((syN, int(self.T/self.dt)), dtype="float32") 
        offset = int(self.w/2)  #ms
        c=0  
        for i, pair in enumerate(pairs):
            s = pair[0]
            post_i = pair[1]
            for tb in range(self.tbs):
                # print(tb)
                rho_file = self.drc_syn+"ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+ "rho"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
                frho= open(rho_file, "rb")
                rectype = np.dtype(np.float64)
                rho_tb = np.fromfile(frho, dtype=rectype).astype("float32")
                # print(rho_tb)
                if tb==0:
                    rho=rho_tb
                else:
                    rho=np.append(rho, rho_tb).astype("float32")
                frho.close()
            
            con_mat[i] = rho
            del rho
            gc.collect()
            
        #A->B
        AB_cons = []
        for pair in pairs:
            if pair[0]<Gex and Gex<=pair[1]<2*Gex:
                AB_cons.append(pair)
                
        #A->A
        AA_cons = []
        for pair in pairs:
            if pair[0]<Gex and pair[1]<Gex:
                AA_cons.append(pair)
                
        #B->A
        BA_cons= []
        for pair in pairs:
            if pair[1]<Gex and Gex<=pair[0]<2*Gex:
                BA_cons.append(pair)
                
        #B->B
        BB_cons= []
        for pair in pairs:
            if Gex<=pair[1]<2*Gex and Gex<=pair[0]<2*Gex:
                BB_cons.append(pair)
                
                
        #B->C
        BC_cons = []
        for pair in pairs:
            if Gex<=pair[0]<2*Gex and 2*Gex<=pair[1]:
                BC_cons.append(pair)
                
        #C->B
        CB_cons = []
        for pair in pairs:
            if 2*Gex<=pair[0] and Gex<=pair[1]<2*Gex:
                CB_cons.append(pair)
                
        #A-> C
        AC_cons = []
        for pair in pairs:
            if pair[0]<Gex and 2*Gex<=pair[1]:
                AC_cons.append(pair)
                
        #C-> C
        CC_cons = []
        for pair in pairs:
            if 2*Gex<=pair[0] and 2*Gex<=pair[1]:
                CC_cons.append(pair)
                
        #C->A
        CA_cons= []
        for pair in pairs:
            if pair[1]<Gex and 2*Gex<=pair[0]:
                CA_cons.append(pair)
            
        
            
        #plot mean rho
        AB_m =np.zeros((len(AB_cons),int(self.T/self.dt)), dtype="float32")
        for n,AB_con in enumerate(AB_cons):
            conin = pairs.index(AB_con)
            AB_m[n]=con_mat[conin]
        AB_mt=np.mean(AB_m, axis=0)
        
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, AB_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('AB_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)
        AB_1_m = (AB_mt[int((stim_start+stim_dur)/self.dt)])
        
        last_AB_m = (AB_mt[-1])
        AB_1_std = (np.std(AB_m, axis=0)/AB_mt)[int((stim_start+stim_dur)/self.dt)]
        last_AB_std = (np.std(AB_m, axis=0)/AB_mt)[-1]
        
        print(last_AB_m)
        del AB_m
        # del AB_mt
        gc.collect()
        
        
        AA_m =np.zeros((len(AA_cons),int(self.T/self.dt)), dtype="float32")
        for n,AA_con in enumerate(AA_cons):
            conin = pairs.index(AA_con)
            AA_m[n]=con_mat[conin]
        AA_mt=np.mean(AA_m, axis=0)
        
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, AA_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('AA_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)
        AA_1_m = (AA_mt[int((stim_start+stim_dur)/self.dt)])
        
        last_AA_m = (AA_mt[-1])
        AA_1_std = (np.std(AA_m, axis=0)/AA_mt)[int((stim_start+stim_dur)/self.dt)]
        last_AA_std = (np.std(AA_m, axis=0)/AA_mt)[-1]
        
        print(last_AA_m)
        del AA_m
        # del AB_mt
        gc.collect()
        
        
        BA_m =np.zeros((len(BA_cons),int(self.T/self.dt)), dtype="float32")
        for n,BA_con in enumerate(BA_cons):
            conin = pairs.index(BA_con)
            BA_m[n]=con_mat[conin]
        BA_mt=np.mean(BA_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, BA_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('BA_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        BA_1_m = (BA_mt[int((stim_start+stim_dur)/self.dt)])
        # BA_2_m = (BA_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_BA_m = (BA_mt[-1])
        BA_1_std = (np.std(BA_m, axis=0)/BA_mt)[int((stim_start+stim_dur)/self.dt)]
        last_BA_std = (np.std(BA_m, axis=0)/BA_mt)[-1]
        del BA_m
        # del BA_mt
        gc.collect()
        
        
        BB_m =np.zeros((len(BB_cons),int(self.T/self.dt)), dtype="float32")
        for n,BB_con in enumerate(BB_cons):
            conin = pairs.index(BB_con)
            BB_m[n]=con_mat[conin]
        BB_mt=np.mean(BB_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, BB_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('BB_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        BB_1_m = (BB_mt[int((stim_start+stim_dur)/self.dt)])
        # BA_2_m = (BA_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_BB_m = (BB_mt[-1])
        BB_1_std = (np.std(BB_m, axis=0)/BB_mt)[int((stim_start+stim_dur)/self.dt)]
        last_BB_std = (np.std(BB_m, axis=0)/BB_mt)[-1]
        del BB_m
        # del BA_mt
        gc.collect()
        
        
        
        BC_m =np.zeros((len(BC_cons),int(self.T/self.dt)), dtype="float32")
        for n,BC_con in enumerate(BC_cons):
            conin = pairs.index(BC_con)
            BC_m[n]=con_mat[conin]
        BC_mt=np.mean(BC_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, BC_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('BC_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        BC_1_m = (BC_mt[int((stim_start+stim_dur)/self.dt)])
        # BC_2_m = (BC_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_BC_m = (BC_mt[-1])
        BC_1_std = (np.std(BC_m, axis=0)/BC_mt)[int((stim_start+stim_dur)/self.dt)]
        last_BC_std = (np.std(BC_m, axis=0)/BC_mt)[-1]
        del BC_m
        # del BC_mt
        gc.collect()
                
        


        CB_m =np.zeros((len(CB_cons),int(self.T/self.dt)), dtype="float32")
        for n,CB_con in enumerate(CB_cons):
            conin = pairs.index(CB_con)
            CB_m[n]=con_mat[conin]
        CB_mt=np.mean(CB_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, CB_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('CB_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        CB_1_m = (CB_mt[int((stim_start+stim_dur)/self.dt)])
        # CB_2_m = (CB_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_CB_m = (CB_mt[-1])
        CB_1_std = (np.std(CB_m, axis=0)/CB_mt)[int((stim_start+stim_dur)/self.dt)]
        last_CB_std = (np.std(CB_m, axis=0)/CB_mt)[-1]
        del CB_m
        # del CB_mt
        gc.collect()
            
            
        AC_m =np.zeros((len(AC_cons),int(self.T/self.dt)), dtype="float32")
        for n,AC_con in enumerate(AC_cons):
            conin = pairs.index(AC_con)
            AC_m[n]=con_mat[conin]
        AC_mt=np.mean(AC_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, AC_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('AC_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        AC_1_m = (AC_mt[int((stim_start+stim_dur)/self.dt)])
        # AC_2_m = (AC_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_AC_m = (AC_mt[-1])
        AC_1_std = (np.std(AC_m, axis=0)/AC_mt)[int((stim_start+stim_dur)/self.dt)]
        last_AC_std = (np.std(AC_m, axis=0)/AC_mt)[-1]
        del AC_m
        # del AC_mt
        gc.collect()
                
     


        CA_m =np.zeros((len(CA_cons),int(self.T/self.dt)), dtype="float32")
        for n,CA_con in enumerate(CA_cons):
            
            conin = pairs.index(CA_con)
            CA_m[n]=con_mat[conin]
        CA_mt=np.mean(CA_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, CA_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('CA_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        CA_1_m = (CA_mt[int((stim_start+stim_dur)/self.dt)])
        last_CA_m = (CA_mt[-1])
        CA_1_std = (np.std(CA_m, axis=0)/CA_mt)[int((stim_start+stim_dur)/self.dt)]
        last_CA_std = (np.std(CA_m, axis=0)/CA_mt)[-1]
        del CA_m
        # del CA_mt 
        gc.collect()
        
        
        CC_m =np.zeros((len(CC_cons),int(self.T/self.dt)), dtype="float32")
        for n,CC_con in enumerate(CC_cons):
            
            conin = pairs.index(CC_con)
            CC_m[n]=con_mat[conin]
        CC_mt=np.mean(CC_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, CC_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('CC_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        CC_1_m = (CC_mt[int((stim_start+stim_dur)/self.dt)])
        last_CC_m = (CC_mt[-1])
        CC_1_std = (np.std(CC_m, axis=0)/CC_mt)[int((stim_start+stim_dur)/self.dt)]
        last_CC_std = (np.std(CC_m, axis=0)/CC_mt)[-1]
        del CC_m
        # del CC_mt 
        gc.collect()
        
        
        #mean weight of stimed pair
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, (AB_mt+BA_mt+AA_mt+BB_mt)/4)
        axes.set_title("mean efficacy of stimulated neurons")
        axes.set_xlabel('Time (s)',fontsize = fs)
        axes.set_ylabel('Synaptic efficacy',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        del AA_mt
        del AB_mt
        del BB_mt
        del BA_mt
        gc.collect()
        
        
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, (AC_mt+CA_mt+CB_mt+BC_mt)/4)
        axes.set_title("mean efficacy of hemi-stimulated neurons")
        axes.set_xlabel('Time (s)',fontsize = fs)
        axes.set_ylabel('Synaptic efficacy',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig) 
        
        del AC_mt
        del CB_mt
        del BC_mt
        del CA_mt
        gc.collect()
    

        pdf.close()
        plt.close()
    
    
        f=open(self.savef + "_last_m.txt", "w")
        f.write("After 1st stim:"+"\n")
        
        f.write("AB:"+str(AB_1_m)+"\n")
        f.write("BA:"+str(BA_1_m)+"\n")
        f.write("BC:"+str(BC_1_m)+"\n")
        f.write("CB:"+str(CB_1_m)+"\n")
        f.write("AC:"+str(AC_1_m)+"\n")
        f.write("CA:"+str(CA_1_m)+"\n")
        # f.write("AD:"+str(AD_1_m)+"\n")
        # f.write("DA:"+str(DA_1_m)+"\n"+"\n")
        
        # f.write("After 2nd stim:"+"\n")
        # f.write("AB:"+str(AB_2_m)+"\n")
        # f.write("BA:"+str(BA_2_m)+"\n")
        # f.write("BC:"+str(BC_2_m)+"\n")
        # f.write("CB:"+str(CB_2_m)+"\n")
        # f.write("AC:"+str(AC_2_m)+"\n")
        # f.write("CA:"+str(CA_2_m)+"\n")
        # f.write("AD:"+str(AD_2_m)+"\n")
        # f.write("DA:"+str(DA_2_m)+"\n"+"\n")
        
        
        f.write("After SWS:"+"\n")
        f.write("AB:"+str(last_AB_m)+"\n")
        f.write("BA:"+str(last_BA_m)+"\n")
        f.write("BC:"+str(last_BC_m)+"\n")
        f.write("CB:"+str(last_CB_m)+"\n")
        f.write("AC:"+str(last_AC_m)+"\n")
        f.write("CA:"+str(last_CA_m)+"\n")
        # f.write("AD:"+str(last_AD_m)+"\n")
        # f.write("DA:"+str(last_DA_m)+"\n")
        f.close()
        
        f=open(self.savef + "_last_cv.txt", "w")
        f.write("After 1st stim:"+"\n")
        
        f.write("AB:"+str(AB_1_std)+"\n")
        f.write("BA:"+str(BA_1_std)+"\n")
        f.write("BC:"+str(BC_1_std)+"\n")
        f.write("CB:"+str(CB_1_std)+"\n")
        f.write("AC:"+str(AC_1_std)+"\n")
        f.write("CA:"+str(CA_1_std)+"\n")
        
        
        f.write("After SWS:"+"\n")
        f.write("AB:"+str(last_AB_std)+"\n")
        f.write("BA:"+str(last_BA_std)+"\n")
        f.write("BC:"+str(last_BC_std)+"\n")
        f.write("CB:"+str(last_CB_std)+"\n")
        f.write("AC:"+str(last_AC_std)+"\n")
        f.write("CA:"+str(last_CA_std)+"\n")
        f.close()
        
        
        
        f2=open(self.savef + "_wratio.txt", "w")
        stim_w_1 = (AB_1_m+BA_1_m)/2
        nostim_w_1 = (AC_1_m+CA_1_m+BC_1_m+CB_1_m)/4
        ratio_1 = stim_w_1/nostim_w_1
        f2.write("ratio_after_stim:{}\n".format(ratio_1))
        stim_w_s = (last_AB_m+last_BA_m)/2
        nostim_w_s = (last_AC_m+last_CA_m+last_CB_m+last_BC_m)/4
        ratio_s = stim_w_s/nostim_w_s
        f2.write("ratio_after_sws:{}\n".format(ratio_s))
        f2.close()
        
        f2=open(self.savef + "_cvratio.txt", "w")
        stim_w_1 = (AB_1_std+BA_1_std)/2
        nostim_w_1 = (AC_1_std+CA_1_std+BC_1_std+CB_1_std)/4
        ratio_1 = stim_w_1/nostim_w_1
        f2.write("ratio_after_stim:{}\n".format(ratio_1))
        stim_w_s = (last_AB_std+last_BA_std)/2
        nostim_w_s = (last_AC_std+last_CA_std+last_CB_std+last_BC_std)/4
        ratio_s = stim_w_s/nostim_w_s
        f2.write("ratio_after_sws:{}\n".format(ratio_s))
        f2.close()
        
        
        ratio_sws_stim = ((last_AB_m+last_BA_m+last_BB_m+last_AA_m)/4) / ((AB_1_m+BA_1_m+BB_1_m+AA_1_m)/4)
        diff_sws_stim = ((last_AB_m+last_BA_m+last_BB_m+last_AA_m)/4) - ((AB_1_m+BA_1_m+BB_1_m+AA_1_m)/4)
        
        ratio_sws_nstim = ((last_AC_m+last_CA_m+last_BC_m+last_CB_m)/4)/((AC_1_m+CA_1_m+BC_1_m+CB_1_m)/4)
        diff_sws_nstim = ((last_AC_m+last_CA_m+last_BC_m+last_CB_m)/4)-((AC_1_m+CA_1_m+BC_1_m+CB_1_m)/4)
        
        ratio_sws_nn= (last_CC_m)/(CC_1_m)
        diff_sws_nn = (last_CC_m)-(CC_1_m)
        
        f3=open(self.savef + "_sws_stim.txt", "w")
        f3.write("ratio_sws_stim:{}\n".format(ratio_sws_stim))
        f3.write("diff_sws_stim:{}\n".format(diff_sws_stim))
        f3.write("ratio_sws_nstim:{}\n".format(ratio_sws_nstim))
        f3.write("diff_sws_nstim:{}\n".format(diff_sws_nstim))
        f3.write("ratio_sws_nn:{}\n".format(ratio_sws_nn))
        f3.write("diff_sws_nn:{}\n".format(diff_sws_nn))
    
    def weights_cal_interval(self, Gex, stim_start, stim_dur, interval):
        pdf = PdfPages(self.savef+"_weights.pdf")
        syN=0
        pairs = []
        for post_i in range(self.NE+self.NI):
            for s in range(self.NE+self.NI):
                cpre_file = self.drc_syn+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+ "cpre"+str(s)+"-"+str(post_i)+"_"+str(0)+".bin"
                if os.path.exists(cpre_file):
                    pairs.append([s,post_i])
                    syN += 1
                    
        con_mat = np.zeros((syN, int(self.T/self.dt)), dtype="float32") 
        offset = int(self.w/2)  #ms
        c=0  
        for i, pair in enumerate(pairs):
            s = pair[0]
            post_i = pair[1]
            for tb in range(self.tbs):
                # print(tb)
                rho_file = self.drc_syn+"ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+ "rho"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
                frho= open(rho_file, "rb")
                rectype = np.dtype(np.float64)
                rho_tb = np.fromfile(frho, dtype=rectype).astype("float32")
                # print(rho_tb)
                if tb==0:
                    rho=rho_tb
                else:
                    rho=np.append(rho, rho_tb).astype("float32")
                frho.close()
            
            con_mat[i] = rho
            del rho
            gc.collect()
            
        #A->B
        AB_cons = []
        for pair in pairs:
            if pair[0]<Gex and Gex<=pair[1]<2*Gex:
                AB_cons.append(pair)
                
        #B->A
        BA_cons= []
        for pair in pairs:
            if pair[1]<Gex and Gex<=pair[0]<2*Gex:
                BA_cons.append(pair)
                
                
        #B->C
        BC_cons = []
        for pair in pairs:
            if Gex<=pair[0]<2*Gex and 2*Gex<=pair[1]<3*Gex:
                BC_cons.append(pair)
                
        #C->B
        CB_cons = []
        for pair in pairs:
            if 2*Gex<=pair[0]<3*Gex and Gex<=pair[1]<2*Gex:
                CB_cons.append(pair)
                
        #A-> C
        AC_cons = []
        for pair in pairs:
            if pair[0]<Gex and 2*Gex<=pair[1]<3*Gex:
                AC_cons.append(pair)
                
        #C->A
        CA_cons= []
        for pair in pairs:
            if pair[1]<Gex and 2*Gex<=pair[0]<3*Gex:
                CA_cons.append(pair)
            
            
        #A-> D
        AD_cons = []
        for pair in pairs:
            if pair[0]<Gex and 3*Gex<=pair[1]<4*Gex:
                AD_cons.append(pair)
                
        #D->A
        DA_cons= []
        for pair in pairs:
            if pair[1]<Gex and 3*Gex<=pair[0]<4*Gex:
                DA_cons.append(pair)
            
        #plot mean rho
        AB_m =np.zeros((len(AB_cons),int(self.T/self.dt)), dtype="float32")
        for n,AB_con in enumerate(AB_cons):
            conin = pairs.index(AB_con)
            AB_m[n]=con_mat[conin]
        AB_mt=np.mean(AB_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, AB_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('AB_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)
        AB_1_m = (AB_mt[int((stim_start+stim_dur)/self.dt)])
        AB_2_m = (AB_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_AB_m = (AB_mt[-1])
        print(last_AB_m)
        del AB_m
        del AB_mt
        gc.collect()
        
        
        BA_m =np.zeros((len(BA_cons),int(self.T/self.dt)), dtype="float32")
        for n,BA_con in enumerate(BA_cons):
            conin = pairs.index(BA_con)
            BA_m[n]=con_mat[conin]
        BA_mt=np.mean(BA_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, BA_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('BA_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        BA_1_m = (BA_mt[int((stim_start+stim_dur)/self.dt)])
        BA_2_m = (BA_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_BA_m = (BA_mt[-1])
        del BA_m
        del BA_mt
        gc.collect()
        
        BC_m =np.zeros((len(BC_cons),int(self.T/self.dt)), dtype="float32")
        for n,BC_con in enumerate(BC_cons):
            conin = pairs.index(BC_con)
            BC_m[n]=con_mat[conin]
        BC_mt=np.mean(BC_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, BC_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('BC_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        BC_1_m = (BC_mt[int((stim_start+stim_dur)/self.dt)])
        BC_2_m = (BC_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_BC_m = (BC_mt[-1])
        del BC_m
        del BC_mt
        gc.collect()
                
        


        CB_m =np.zeros((len(CB_cons),int(self.T/self.dt)), dtype="float32")
        for n,CB_con in enumerate(CB_cons):
            conin = pairs.index(CB_con)
            CB_m[n]=con_mat[conin]
        CB_mt=np.mean(CB_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, CB_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('CB_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        CB_1_m = (CB_mt[int((stim_start+stim_dur)/self.dt)])
        CB_2_m = (CB_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_CB_m = (CB_mt[-1])
        del CB_m
        del CB_mt
        gc.collect()
            
            
        AC_m =np.zeros((len(AC_cons),int(self.T/self.dt)), dtype="float32")
        for n,AC_con in enumerate(AC_cons):
            conin = pairs.index(AC_con)
            AC_m[n]=con_mat[conin]
        AC_mt=np.mean(AC_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, AC_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('AC_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        AC_1_m = (AC_mt[int((stim_start+stim_dur)/self.dt)])
        AC_2_m = (AC_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_AC_m = (AC_mt[-1])
        del AC_m
        del AC_mt
        gc.collect()
                
     


        CA_m =np.zeros((len(CA_cons),int(self.T/self.dt)), dtype="float32")
        for n,CA_con in enumerate(CA_cons):
            
            conin = pairs.index(CA_con)
            CA_m[n]=con_mat[conin]
        CA_mt=np.mean(CA_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, CA_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('CA_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        CA_1_m = (CA_mt[int((stim_start+stim_dur)/self.dt)])
        CA_2_m = (CA_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_CA_m = (CA_mt[-1])
        del CA_m
        del CA_mt
        gc.collect()
        
        AD_m =np.zeros((len(AD_cons),int(self.T/self.dt)), dtype="float32")
        for n,AD_con in enumerate(AD_cons):
            conin = pairs.index(AD_con)
            AD_m[n]=con_mat[conin]
        AD_mt=np.mean(AD_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, AD_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('AD_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        AD_1_m = (AD_mt[int((stim_start+stim_dur)/self.dt)])
        AD_2_m = (AD_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_AD_m = (AD_mt[-1])
        del AD_m
        del AD_mt
        gc.collect()
                

        DA_m =np.zeros((len(DA_cons),int(self.T/self.dt)), dtype="float32")
        for n,DA_con in enumerate(DA_cons):
            
            conin = pairs.index(DA_con)
            DA_m[n]=con_mat[conin]
        DA_mt=np.mean(DA_m, axis=0)
        fig, axes = plt.subplots(figsize=(20, 8))
        axes.plot(self.x, DA_mt)
        axes.set_xlabel('Time (sec)',fontsize = fs)
        axes.set_ylabel('CA_weight_mean',fontsize = fs)
        axes.tick_params(labelsize=fs_l)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(0,)
        axes.set_rasterized(True)
        pdf.savefig(fig)  
        DA_1_m = (DA_mt[int((stim_start+stim_dur)/self.dt)])
        DA_2_m = (DA_mt[int((stim_start+interval+2*stim_dur)/self.dt)])
        last_DA_m = (DA_mt[-1])
        del DA_m
        del DA_mt
        gc.collect()

        pdf.close()
        plt.close()
    
    
        f=open(self.savef + "_last_m.txt", "w")
        f.write("After 1st stim:"+"\n")
        
        f.write("AB:"+str(AB_1_m)+"\n")
        f.write("BA:"+str(BA_1_m)+"\n")
        f.write("BC:"+str(BC_1_m)+"\n")
        f.write("CB:"+str(CB_1_m)+"\n")
        f.write("AC:"+str(AC_1_m)+"\n")
        f.write("CA:"+str(CA_1_m)+"\n")
        f.write("AD:"+str(AD_1_m)+"\n")
        f.write("DA:"+str(DA_1_m)+"\n"+"\n")
        
        f.write("After 2nd stim:"+"\n")
        f.write("AB:"+str(AB_2_m)+"\n")
        f.write("BA:"+str(BA_2_m)+"\n")
        f.write("BC:"+str(BC_2_m)+"\n")
        f.write("CB:"+str(CB_2_m)+"\n")
        f.write("AC:"+str(AC_2_m)+"\n")
        f.write("CA:"+str(CA_2_m)+"\n")
        f.write("AD:"+str(AD_2_m)+"\n")
        f.write("DA:"+str(DA_2_m)+"\n"+"\n")
        
        
        f.write("After SWS:"+"\n")
        f.write("AB:"+str(last_AB_m)+"\n")
        f.write("BA:"+str(last_BA_m)+"\n")
        f.write("BC:"+str(last_BC_m)+"\n")
        f.write("CB:"+str(last_CB_m)+"\n")
        f.write("AC:"+str(last_AC_m)+"\n")
        f.write("CA:"+str(last_CA_m)+"\n")
        f.write("AD:"+str(last_AD_m)+"\n")
        f.write("DA:"+str(last_DA_m)+"\n")
        
        
        f.close()
        
        
    
    def rho_cal_multi(self, cores, sleep_durs, wake_durs):
        args = []
        syN=0
        N_con =[]
        manager = Manager()
        
        for post_i in range(self.NE+self.NI):
            for s in range(self.NE+self.NI):
                cpre_file = self.drc_syn+  "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"cpre"+str(s)+"-"+str(post_i)+"_"+str(0)+".bin"
                if os.path.exists(cpre_file):
                    syN += 1
                    N_con.append([s,post_i])
                    
        self.N_rho_s_T = manager.Array('d', range(syN))
        self.N_rho_w_T = manager.Array('d', range(syN))
        self.k_min_li = manager.Array('d', range(syN))
    
        for i in range(syN):
            args.append((i, N_con[i], sleep_durs, wake_durs))
        print(f'analyze networks: using {cores} cores')
        with Pool(processes=cores) as pool:
            pool.map(self.rho_sw_cal, args)
    
    
    def rho_sw_cal(self, args):
        i, N_con, sleep_durs, wake_durs = args 
        s=N_con[0]
        post_i=N_con[1]
        for tb in range(self.tbs):
            rho_file = self.drc_syn+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"rho"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
            frho= open(rho_file, "rb")
            rectype = np.dtype(np.float64)
            rho_tb = np.fromfile(frho, dtype=rectype).astype("float32")
            if tb==0:
                rho=rho_tb
            else:
                rho=np.append(rho, rho_tb).astype("float32")
            frho.close()
            
        
        offset = int(self.w/2)  #ms
        if len(sleep_durs)!=0:
            # print("get sleep rho")
            m_rho_s = 0
            for n, dur_s in enumerate(sleep_durs):
                if dur_s[0]==0:
                    m_rho_s += np.mean(rho[dur_s[0]+int(offset/self.dt):dur_s[1] +int(offset/self.dt)])
                else:
                    m_rho_s += np.mean(rho[dur_s[0]+int(offset/self.dt):dur_s[1] +int(offset/self.dt)])
                # print("m_rho_s", np.mean(rho[dur_s[0]:dur_s[1]]))
            # print("m_rho_s", m_rho_s)
            self.N_rho_s_T[i]=(m_rho_s/len(sleep_durs))
        
        if len(wake_durs)!=0:
            m_rho_w = 0
            for n,dur_w in enumerate(wake_durs):
                if dur_w[0]==0:
                    m_rho_w += np.mean(rho[dur_w[0]+int(offset/self.dt):dur_w[1] +int(offset/self.dt)])
                else:
                    m_rho_w += np.mean(rho[dur_w[0]+int(offset/self.dt):dur_w[1] +int(offset/self.dt)])
            self.N_rho_w_T[i]=m_rho_w/len(wake_durs)
            
        del rho
        gc.collect()
            
        # for tb in range(self.tbs):
        #     k_file = self.drc_syn+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"K"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
        #     fk= open(k_file, "rb")
        #     rectype = np.dtype(np.float64)
        #     k_tb = np.fromfile(fk, dtype=rectype).astype("float32")
        #     if tb==0:
        #         k=k_tb
        #     else:
        #         k=np.append(k, k_tb).astype("float32")
        #     fk.close()
        # self.k_min_li[i] = np.min(k[int(self.T/2/self.dt):])
            
    def rho_col(self):
        print(self.N_rho_s_T)
        nonzero_i=np.nonzero(np.array(self.N_rho_s_T))[0]
        # print("nonzero index", nonzero_i)
        nonzero_i_w=np.nonzero(np.array(self.N_rho_w_T))[0]
        N_rho_s_T = np.take(np.array(self.N_rho_s_T), nonzero_i)
        N_rho_w_T = np.take(np.array(self.N_rho_w_T), nonzero_i_w)
        # print("s_Total", N_rho_s_T)
        # print("w_Total", N_rho_w_T)
        if len(N_rho_s_T)!=0 and len(N_rho_w_T)!=0:
            m_rho_s_T = np.mean(N_rho_s_T)
            m_rho_w_T = np.mean(N_rho_w_T)
            m_rho_np = np.array([m_rho_s_T, m_rho_w_T])
            
        else:
            m_rho_np = np.array([0, 0])
            
        print("mean_rho_s,mean_rho_w", m_rho_np)
        np.save(self.savef+"_rhosw",m_rho_np)
        
        # k_min_m = np.mean(self.k_min_li)
        # print("k_min_m", k_min_m)
        # np.savetxt(self.savef+"_kmin.csv", k_min_m, delimiter=",")
    
    def rho_sw(self, sleep_durs, wake_durs):
        m_rho_s_T=0
        m_cv_s_T=0
        m_rho_w_T=0
        m_cv_w_T=0
        syN=0
        N_con = []
        for post_i in range(self.NE+self.NI):
            for s in range(self.NE+self.NI):
                cpre_file = self.drc_syn+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+ "cpre"+str(s)+"-"+str(post_i)+"_"+str(0)+".bin"
                if os.path.exists(cpre_file):
                    N_con.append([s,post_i])
                    syN += 1
                    
        rho_li = np.zeros((syN, int(self.T/self.dt)), dtype="float32") 
        offset = int(self.w/2)  #ms
        c=0  
        for i, pair in enumerate(N_con):
            s = pair[0]
            post_i = pair[1]
            for tb in range(self.tbs):
                # print(tb)
                rho_file = self.drc_syn+"ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+ "rho"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
                frho= open(rho_file, "rb")
                rectype = np.dtype(np.float64)
                rho_tb = np.fromfile(frho, dtype=rectype).astype("float32")
                # print(rho_tb)
                if tb==0:
                    rho=rho_tb
                else:
                    rho=np.append(rho, rho_tb).astype("float32")
                frho.close()
            
            rho_li[i] = rho
            del rho
            gc.collect()
            
            
        if len(sleep_durs)!=0:
            print("get sleep rho")
            m_rho_s = 0
            m_cv_s = 0
            for n, dur_s in enumerate(sleep_durs):
                rho_dur_s = rho_li[:, dur_s[0]+int(offset/self.dt):dur_s[1] +int(offset/self.dt)]
                m_cv_s +=np.mean(np.std(rho_dur_s, axis =0)/np.mean(rho_dur_s, axis =0))
                m_rho_s += np.mean(rho_dur_s)
                # print("m_rho_s", np.mean(rho[dur_s[0]:dur_s[1]]))
            print("m_rho_s", m_rho_s)
            m_rho_s_T=(m_rho_s/len(sleep_durs))
            m_cv_s_T=(m_cv_s/len(sleep_durs))
        
        
        if len(wake_durs)!=0:
            # print("get sleep rho")
            m_rho_w = 0
            m_cv_w = 0
            for n, dur_w in enumerate(wake_durs):
                rho_dur_w = rho_li[:, dur_w[0]+int(offset/self.dt):dur_w[1] +int(offset/self.dt)]
                m_cv_w += np.mean(np.std(rho_dur_w, axis =0)/np.mean(rho_dur_w, axis =0))
                m_rho_w += np.mean(rho_dur_w)
                # print("m_rho_s", np.mean(rho[dur_s[0]:dur_s[1]]))
            # print("m_rho_w", m_rho_w)
            m_rho_w_T=(m_rho_w/len(wake_durs))
            m_cv_w_T=(m_cv_w/len(wake_durs))
        del rho_li
        gc.collect()
            
        if  m_rho_s_T!=0 and  m_rho_w_T!=0:
            m_rho_np = np.array([m_rho_s_T, m_rho_w_T])
        else:
            m_rho_np = np.array([0, 0])
            
        if  m_cv_s_T!=0 and  m_cv_w_T!=0:
            m_cv_np = np.array([m_cv_s_T, m_cv_w_T])
        else:
            m_cv_np = np.array([0, 0])
            
            
        print("mean_rho_s,mean_rho_w", m_rho_np)
        np.save(self.savef+"_rhosw",m_rho_np)
        
        print("mean_cv_s,mean_cv_w", m_cv_np)
        np.save(self.savef+"_cvsw",m_cv_np)
    
    
    def graph_saving(self,pdf_save=False, graph_show=False):
            if pdf_save ==True:
                pdf = PdfPages(self.savef+".pdf")
             
            if pdf_save ==True or graph_show ==True:
                for n in range(self.N):
                    try:
                        if self.N > 60 and self.N < 200:
                            if n%2 ==0:
                                for tb in range(self.tbs):
                                    v_file = self.drc_neu+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin"  
                                    f2= open(v_file, "rb")
                                    rectype = np.dtype(np.float64)
                                    v_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                                    if tb==0:
                                        v=v_tb
                                    else:
                                        v=np.append(v, v_tb)
                                    f2.close()
                                if graph_show ==True:
                                    plt.figure(figsize=(15, 4))
                                    plt.title("Neuron{}".format(n))
                                    plt.plot(self.x, v)   #time[0:int(T/dt)-1],
                                    plt.ylim(-100, 40)
                                    plt.xlabel('Time (sec)',fontsize = fs)
                            #             plt.xlim(0,2)
                                        #plt.ylim(0,1)
                                    plt.xticks(fontsize = fs)
                                    plt.yticks(fontsize = fs)

                                    plt.show()
                                if pdf_save==True:
                                    fig, axes = plt.subplots(figsize=(20, 8))
                                    axes.set_title("param {}\n, Neuron{} ex{}".format(self.date_i, n,self.ex))
                                   
                                    axes.plot(self.x, v)   #time[0:int(T/dt)-1],
                                    axes.set_xlabel('Time (sec)',fontsize = fs)
                                    axes.tick_params(labelsize = fs_l)

                                    axes.set_ylim(-110,40)
                                    axes.set_rasterized(True)
                                    pdf.savefig(fig)
                                    plt.close()
                                
                                
                                cpre, cpost, rho, s = self.out_syn(1, n, self.drc_syn)
                                fig, axes = plt.subplots(figsize=(20, 8))
                                axes.set_title("Ca2+, syn{}-{}".format(s,n))
                                axes.plot(self.x, cpre + cpost)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Ca2+',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()  
                                
                                fig, axes = plt.subplots(figsize=(20, 8))
                                axes.set_title("cpre, syn{}-{}".format(s,n))
                                axes.plot(self.x, cpre)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Ca2+_pre',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()  
                                
                                fig, axes = plt.subplots(figsize=(20, 8))
                                axes.set_title("cpost, syn{}-{}".format(s,n))
                                axes.plot(self.x, cpost)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Ca2+_post',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()  
                                
                                del cpre
                                del cpost
                                gc.collect()
                                
                                fig, axes = plt.subplots(figsize=(20, 8))
                                axes.set_title("rho, syn{}-{}".format(s,n))
                                axes.plot(self.x, rho)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Efficacy',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                axes.set_ylim(0, 1.05)
                                pdf.savefig(fig)
                                plt.close()  
                                del rho
                                gc.collect()
                                
        #                         fig, axes = plt.subplots(figsize=(20, 8))
        # #                             axes.set_title("cpost, Neuron{}".format(n))
        #                         axes.plot(self.x,Sp, label="r")   #time[0:int(T/dt)-1],
        #                         axes.plot(self.x, K, label="a")
        #                         axes.set_xlabel('Time (sec)',fontsize = fs)
        #                         # axes.set_ylabel('p-CaMK2',fontsize = fs)
        #                         axes.set_ylim(0,1.05)
        #                         axes.tick_params(labelsize = fs_l)
        #                         axes.set_rasterized(True)
        #                         axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize=fs)
        #                         pdf.savefig(fig)
        #                         plt.close()  
        #                         del Sp
        #                         del K
        #                         gc.collect()
                                
        #                         fig, axes = plt.subplots(figsize=(20, 8))
        # #                             axes.set_title("cpost, Neuron{}".format(n))
        #                         axes.plot(K)   #time[0:int(T/dt)-1],
        #                         axes.set_xlabel('Time (sec)',fontsize = fs)
        #                         axes.set_ylabel('Phosphatase',fontsize = fs)
        #                         axes.tick_params(labelsize = fs_l)
        #                         axes.set_rasterized(True)
        #                         pdf.savefig(fig)
        #                         plt.close()  
        #                         del K
        #                         gc.collect()
                                
                                # if "_i" in self.model:
                                #     for tb in range(self.tbs):
                                #         ia_file = self.drc_neu+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"iampar"+str(n)+"_"+str(tb)+".bin" 
                                #         f2= open(ia_file, "rb")
                                #         rectype = np.dtype(np.float64)
                                #         ia_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                                #         if tb==0:
                                #             ia=ia_tb
                                #         else:
                                #             ia=np.append(ia, ia_tb)
                                #         f2.close()
                                        
                                #     fig, axes = plt.subplots(figsize=(20, 8))
                                #     # axes.set_title("param {}\n, Neuron{} ex{}".format(self.date_i, n,self.ex))
                                   
                                #     axes.plot(ia)   #time[0:int(T/dt)-1],
                                #     axes.set_xlabel('Time (sec)',fontsize = fs)
                                #     axes.set_ylabel('i_ampar (nA)',fontsize = fs)
                                #     # axes.set_ylim(-110,40)
                                #     # axes.set_ylim(-110,40)
                                #     axes.tick_params(labelsize = fs_l)
                                #     axes.set_rasterized(True)
                                #     pdf.savefig(fig)
                                #     plt.close() 
                                #     del ia
                                #     gc.collect() 
                                    
                                #     for tb in range(self.tbs):
                                #         iall_file = self.drc_neu+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"iall"+str(n)+"_"+str(tb)+".bin" 
                                #         f2= open(iall_file, "rb")
                                #         rectype = np.dtype(np.float64)
                                #         iall_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                                #         if tb==0:
                                #             iall=iall_tb
                                #         else:
                                #             iall=np.append(iall, iall_tb)
                                #         f2.close()
                                        
                                #     fig, axes = plt.subplots(figsize=(20, 8))
                                #     # axes.set_title("param {}\n, Neuron{} ex{}".format(self.date_i, n,self.ex))
                                   
                                #     axes.plot(iall)   #time[0:int(T/dt)-1],
                                #     axes.set_xlabel('Time (sec)',fontsize = fs)
                                #     axes.set_ylabel('i_ampar (nA)',fontsize = fs)
                                #     # axes.set_ylim(-110,40)
                                #     # axes.set_ylim(-110,40)
                                #     axes.tick_params(labelsize = fs_l)
                                #     axes.set_rasterized(True)
                                #     pdf.savefig(fig)
                                #     plt.close() 
                                #     del iall
                                #     gc.collect() 

                               
                                
                                
                        elif self.N > 200:
                            if n%25 ==0:
                                for tb in range(self.tbs):
                                    v_file = self.drc_neu+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin"  
                                    f2= open(v_file, "rb")
                                    rectype = np.dtype(np.float64)
                                    v_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                                    if tb==0:
                                        v=v_tb
                                    else:
                                        v=np.append(v, v_tb)
                                    f2.close()
                                if graph_show ==True:
                                    plt.figure(figsize=(15, 4))
                                    plt.title("Neuron{}".format(n))
                                    plt.plot(self.x, v)   #time[0:int(T/dt)-1],
                                    plt.ylim(-100, 40)
                                    plt.xlabel('Time (sec)',fontsize = fs)
                            #             plt.xlim(0,2)
                                        #plt.ylim(0,1)
                                    plt.xticks(fontsize = fs)
                                    plt.yticks(fontsize = fs)

                                    plt.show()
                                if pdf_save==True:
                                    fig, axes = plt.subplots(figsize=(20, 8))
                                    axes.set_title("param {}\n, Neuron{} ex{}".format(self.date_i, n,self.ex))
                                   
                                    axes.plot(self.x, v)   #time[0:int(T/dt)-1],
                                    axes.set_xlabel('Time (sec)',fontsize = fs)
                                    axes.tick_params(labelsize = fs_l)

                                    axes.set_ylim(-110,40)
                                    axes.set_rasterized(True)
                                    pdf.savefig(fig)
                                    plt.close()
                                
                                
                                cpre, cpost, rho,s = self.out_syn(1, n, self.drc_syn)
                                fig, axes = plt.subplots(figsize=(20, 8))
        #                             axes.set_title("cpost, Neuron{}".format(n))
                                axes.plot(self.x, cpre + cpost)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Ca2+',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()  
                                
                                fig, axes = plt.subplots(figsize=(20, 8))
        #                             axes.set_title("cpost, Neuron{}".format(n))
                                axes.plot(self.x, cpre)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Ca2+_pre',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()  
                                
                                fig, axes = plt.subplots(figsize=(20, 8))
        #                             axes.set_title("cpost, Neuron{}".format(n))
                                axes.plot(self.x, cpost)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Ca2+_post',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()  
                                
                                del cpre
                                del cpost
                                gc.collect()
                                
                                fig, axes = plt.subplots(figsize=(20, 8))
        #                             axes.set_title("cpost, Neuron{}".format(n))
                                axes.plot(self.x, rho)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Efficacy',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                axes.set_ylim(0, 1.05)
                                pdf.savefig(fig)
                                plt.close()  
                                del rho
                                gc.collect()
                                
                                
                        else:
                            if n%1==0:
                                for tb in range(self.tbs):
                                    v_file = self.drc_neu+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin"  
                                    f2= open(v_file, "rb")
                                    rectype = np.dtype(np.float64)
                                    v_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                                    if tb==0:
                                        v=v_tb
                                    else:
                                        v=np.append(v, v_tb)
                                    f2.close()
                                if graph_show ==True:
                                    plt.figure(figsize=(15, 4))
                                    plt.title("Neuron{}".format(n))
                                    plt.plot(self.x, v)   #time[0:int(T/dt)-1],
                                    plt.ylim(-100, 40)
                                    plt.xlabel('Time (sec)',fontsize = fs)
                            #             plt.xlim(0,2)
                                        #plt.ylim(0,1)
                                    plt.xticks(fontsize = fs)
                                    plt.yticks(fontsize = fs)

                                    plt.show()
                                if pdf_save==True:
                                    fig, axes = plt.subplots(figsize=(20, 8))
                                    axes.set_title("param {}\n, Neuron{} ex{}".format(self.date_i, n,self.ex))
                                   
                                    axes.plot(self.x, v)   #time[0:int(T/dt)-1],
                                    axes.set_xlabel('Time (sec)',fontsize = fs)
                                    axes.tick_params(labelsize = fs_l)

                                    axes.set_ylim(-110,40)
                                    axes.set_rasterized(True)
                                    pdf.savefig(fig)
                                    plt.close()
                                
                                
                                cpre, cpost, rho,s = self.out_syn(1, n, self.drc_syn)
                                fig, axes = plt.subplots(figsize=(20, 8))
        #                             axes.set_title("cpost, Neuron{}".format(n))
                                axes.plot(self.x, cpre + cpost)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Ca2+',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()  
                                
                                fig, axes = plt.subplots(figsize=(20, 8))
        #                             axes.set_title("cpost, Neuron{}".format(n))
                                axes.plot(self.x, cpre)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Ca2+_pre',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()  
                                
                                fig, axes = plt.subplots(figsize=(20, 8))
        #                             axes.set_title("cpost, Neuron{}".format(n))
                                axes.plot(self.x, cpost)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Ca2+_post',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()  
                                
                                del cpre
                                del cpost
                                gc.collect()
                                
                                fig, axes = plt.subplots(figsize=(20, 8))
        #                             axes.set_title("cpost, Neuron{}".format(n))
                                axes.plot(self.x, rho)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)',fontsize = fs)
                                axes.set_ylabel('Efficacy',fontsize = fs)
                                axes.tick_params(labelsize = fs_l)
                                axes.set_rasterized(True)
                                axes.set_ylim(0, 1.05)
                                pdf.savefig(fig)
                                plt.close()  
                                del rho
                                gc.collect()
                                
                                        
                    except:
                        traceback.print_exc()
            if pdf_save ==True:
                pdf.close()
                print("pdf_saving_end: {}".format(self.savef))
                
                
    def out_syn(self, pre_n, post_i, drc_syn):
    #         ex_in = ex+self.ex_diff
        c=0
        for s in range(self.N):
            cpre_file = drc_syn+  "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"cpre"+str(s)+"-"+str(post_i)+"_"+str(0)+".bin"
            if os.path.exists(cpre_file):
#                 print(cpre_file)
               
                for tb in range(self.tbs):
#                     print("tb", tb)
                    cpre_file =drc_syn+  "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"cpre"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
        #                    
                    fcpre= open(cpre_file, "rb")
                    rectype = np.dtype(np.float64)
                    cpre_tb = np.fromfile(fcpre, dtype=rectype).astype("float32")
                    if tb==0:
                        cpre=cpre_tb
                    else:
                        cpre=np.append(cpre, cpre_tb).astype("float32")
                    fcpre.close()
                
                for tb in range(self.tbs):
#                     print("tb", tb)
                    cpost_file =drc_syn+  "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"cpost"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
                    fcpost= open(cpost_file, "rb")
                    rectype = np.dtype(np.float64)
                    cpost_tb = np.fromfile(fcpost, dtype=rectype).astype("float32")
                    if tb==0:
                        cpost=cpost_tb
                    else:
                        cpost=np.append(cpost, cpost_tb).astype("float32")
                    fcpost.close()
                    
                for tb in range(self.tbs):
                    rho_file = drc_syn+ "ex"+str(self.ex)+"_"+"ex_in"+str(self.ex_in)+"_"+"rho"+str(s)+"-"+str(post_i)+"_"+str(tb)+".bin"
                    frho= open(rho_file, "rb")
                    rectype = np.dtype(np.float64)
                    rho_tb = np.fromfile(frho, dtype=rectype).astype("float32")
                    if tb==0:
                        rho=rho_tb
                    else:
                        rho=np.append(rho, rho_tb).astype("float32")
                    frho.close()
                
               
                
                c+=1
                if c==pre_n:
                    break
        return cpre, cpost,rho, s
        
        
    # def multi_cal(self, cores, exs, cv_w, moving, v_off, ISI_M_r, counts, T_w="None", stat = False, raster=False):
    #     args = []
    #     if T_w == "None":
    #         T = self.T
    #     else:
    #         T = T_w
            
    #     if stat ==True:
    #         dt = 0.01
    #         p=os.getcwd()  #/net_search
    #         try:
    #             for i in range(counts):
    #                 cv_dis=np.load(p+"/CV_random_dis/ISIr_{}_{}/N{}_T{}_dt{}/CV_w_{}.npy".format(ISI_M_r[0],ISI_M_r[1], self.N, T, dt, i)).astype('float32')
    #                 if i==0:
    #                     cv_dis_all = cv_dis
    #                 else:
    #                     cv_dis_all = np.concatenate([cv_dis_all, cv_dis]).astype('float32')
                
    #             self.cv_dis_all = np.array(sorted(cv_dis_all))  # small-> large
    #         except:
    #             traceback.print_exc()
    #             self.cv_dis_all = np.zeros(1)
    #     for num, ex in enumerate(exs):
    #         args.append((num, ex, T, cv_w, moving, v_off, stat, raster))
    #     print(f'analyze networks: using {cores} cores')
    #     with Pool(processes=cores) as pool:
    #         pool.map(self.calc_values, args)


    # def get_cv_pv(self, cv):
    #     if cv >= np.max(self.cv_dis_all):
    #         pv = 0
    #     else:
    #         minx = np.argmin(np.abs(self.cv_dis_all-cv))

    #         low_p = minx/len(self.cv_dis_all)*100
    #         up_p = (len(self.cv_dis_all)-minx)/len(self.cv_dis_all)*100
    #         if low_p<up_p:
    #             pv=low_p
    #             # print(low_p)
    #         else:
    #             pv=up_p
    #             # print(up_p)
    #         print("p_value", pv)
    #     return pv
        
    
    def fre_spike(self, v, T, dt):
    
        if np.any(np.isinf(v)) or np.any(np.isnan(v)):
            print("fre_spike: nan")
            return "Excluded",0, 0, np.zeros(int(T/dt), dtype="int8")
        else:
            wavech =WaveCheck(1000)
            pattern = wavech.pattern(v, T, dt, pr=False)
            detv: np.ndarray = signal.detrend(v)
            max_potential: float = max(detv)
            f: np.ndarray  # Array of sample frequencies
            spw: np.ndarray  # Array of power spectral density or power spectrum
            f, spw = periodogram(detv, fs=1/dt*1000)
            maxamp: float = max(spw)
            nummax: int = spw.tolist().index(maxamp)
            maxfre: float = f[nummax]

    #         ntraverse: int = 0
    #         ms = 1
    #         for k in range(len(v)-1):
    #             if k+ms < len(v)-1:
    #                 if (v[k]+20) * (v[k+ms]+20) < 0:
    #                     ntraverse += 1
    #         nspike= (ntraverse/2)/(T/(2*1000))/20
            peaks = find_peaks(v, height=-20)[0]
    #         print(peaks)
            peak_np = np.zeros(int(T/dt), dtype="int8")
            peak_np[peaks]=1
            return pattern.name, maxfre, len(peaks), peak_np
        
# class Down_Judge_Null:
#     def __init__(self, freq, T_s, start, end):
#         self.freq = freq
#         self.T = T_s
#         self.start = start
#         self.end = end

#     def down_judge(self,pars, san_bool):
#         san_x = models.Xmodel(san_bool,0.54)
#         san_x.set_params(pars)
#         s, info = san_x.run_odeint_null(self.freq, self.T_s) #100000, 10

#         h_max=np.max(s[self.start:self.end,2])
#         h_min=np.min(s[self.start:self.end,2])
#         s_max=np.max(s[self.start:self.end,3])
#         s_min=np.min(s[self.start:self.end,3])
#         ca_max=np.max(s[self.start:self.end,4])
#         ca_min=np.min(s[self.start:self.end,4])
# #         print("h_max", h_max)
# #         print("h_min", h_min)
# #         print("s_max", s_max)
# #         print("s_min", s_min)
# #         print("ca_max", ca_max)
# #         print("ca_min", ca_min)

#         init = [-80, 0]
#         h_range = np.linspace(h_min, h_max, 3)
#         # print(h_range)
#         s_range = np.linspace(s_min, s_max+1, 3)
#         ca_range = np.linspace(ca_min, ca_max+1, 3)
#         flag = 0
#         #is there stable node in nk=0 ?
#         for h in h_range:
#             for s_gate in s_range:
#                 for ca in ca_range:
#                     sol = san_x.get_fixp_K5(init, h, ca, s_gate, 1.0e-5, 10000)
#                     print(sol)
#                     if sol[1] < 0.01:
#                         print("possible stable node")
#                         flag = 1
#                         break
#                 if flag == 1:
#                     break
#             if flag == 1:
#                     break

#         #do v, nk orbit go to the stable node?
#         down_flag = 0
#         for h in h_range:
#             for s_gate in s_range:
#                 for ca in ca_range:
#                     args = [h, s_gate, ca]
#                     s_ca, info  = san_x.run_odeint_ca_K5(args, self.freq, self.T_s)
#                     #     print("init", s_ca[0][1])
#                     v_ca_np = s_ca[self.start:self.end,0]
#                     n_ca_np = s_ca[self.start:self.end,1]

#                     vmin_index = np.argmin(v_ca_np)
#                     print(n_ca_np[vmin_index])
#                     if n_ca_np[vmin_index] < 0.01:
#                         print("down")
#                         down_flag = 1
#                         break
#                 if down_flag == 1:
#                     break
#             if down_flag == 1:
#                     break

#         return down_flag
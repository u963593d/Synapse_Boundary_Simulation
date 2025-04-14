# -*- coding: utf-8 -*-




import os
import gc
import sys


from enum import Flag, auto
import numpy as np
from scipy.signal import periodogram, find_peaks
from scipy import signal
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
        
#         if down_j = True:
#             down.down_judge(pars
        
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
    def __init__(self,model, date_i,NE,NI,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_ex, con_in_in,T, Tp, con_log, drc, savedir,exs, ex_diff):
        manager = Manager()
        
        self.date_i = date_i
        self.N = NE + NI
        self.NE = NE
        self.NI = NI
        self.exs = exs
        self.ex_diff = ex_diff
        self.tbs = int(T/Tp)
        self.con_M = con_M
        self.con_ex_ex = con_ex_ex
        self.con_ex_in = con_ex_in
        self.con_in_ex = con_in_ex
        self.con_M_in = con_M_in
        self.con_in_in=con_in_in
      
        self.T = T
        self.Tp=Tp
        
        # self.con_log = con_log
        self.drc = drc
        
        # self.fig_dir = fig_dir
        # self.x = np.arange(0, T, dt)/1000
        
#         self.df_np = np.zeros((len(exs),13))
        self.df_np = manager.Array('d', range(len(exs)*14))

        
        try:
            os.makedirs(savedir+"/"+model+"/")
        except FileExistsError:
            print("{} is already exist".format(savedir+"/"+model+"/"))
            
        if con_log == "True":
            self.savef = savedir+"/{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}".format(model, self.date_i, self.NE,self.NI,self.con_M,self.con_ex_ex,self.con_M_in, self.con_in_in,self.T)
        else:
            self.savef = savedir+"/{}/{}_NE{}_NI{}_conth{}_{}_{}_{}_T{}".format(model, self.date_i, self.NE,self.NI,self.con_ex_ex,self.con_ex_in,self.con_in_ex, self.con_in_in,self.T)
    
    
    def graph_saving(self, exs, pdf_save=False, graph_show=False):
        if pdf_save ==True:
            pdf = PdfPages(self.savef+".pdf")
            
        for num, ex in enumerate(exs):
            if type(ex) != str:
                ex = round(ex,2)
                ex_in = round((ex + self.ex_diff), 2)
            else:
                ex_in = float(ex) +self.ex_diff
                ex_in = f"{ex_in:,.2f}"
            print("exp", ex)

         
            
            try:
                v_file = self.drc+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(0)+"_"+str(0)+".bin" 
                f2= open(v_file, "rb")
                rectype = np.dtype(np.float64)
                v_tb = np.fromfile(f2, dtype=rectype)
                dt = round(self.Tp/v_tb.shape[0],4)
                print("ex {}, dt{}".format(ex, dt))
                self.x = np.arange(0, self.T, dt)/1000
            except:
                print("file_Error, ex{}".format(ex))
                traceback.print_exc()
#             sp_c_li = []
            if pdf_save ==True or graph_show ==True:
                for n in range(self.N):
                    try:
                        if self.N > 60 and self.N < 200:
                            if n%10 ==0:
                                for tb in range(self.tbs):
                                    v_file = self.drc+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
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
                                    plt.plot(v)   #time[0:int(T/dt)-1],
                                    plt.ylim(-100, 40)
                                    plt.xlabel('Time (sec)',fontsize = fs)
                            #             plt.xlim(0,2)
                                        #plt.ylim(0,1)
                                    plt.xticks(fontsize = fs)
                                    plt.yticks(fontsize = fs)

                                    plt.show()
                                if pdf_save==True:
                                    fig, axes = plt.subplots(figsize=(20, 8))
                                    axes.set_title("param {}\n ex {}, Neuron{}".format(self.date_i, ex, n))
                                    axes.plot(v)   #time[0:int(T/dt)-1],
                                    axes.set_xlabel('Time (sec)')
                                    axes.set_ylim(-110,40)
                                    axes.set_rasterized(True)
                                    pdf.savefig(fig)
                                    plt.close()
                                    del v 
                                    gc.collect()
                                # print("max_fre", maxfre)
                                # print("sp_num", sp_c)
                                # print(pattern)
                        elif self.N > 200:
                            if n%200 ==0:
                                for tb in range(self.tbs):
                                    v_file = self.drc+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
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
                                    plt.plot(v)   #time[0:int(T/dt)-1],
                                    plt.ylim(-100, 40)
                                    plt.xlabel('Time (sec)',fontsize = fs)
                            #             plt.xlim(0,2)
                                        #plt.ylim(0,1)
                                    plt.xticks(fontsize = fs)
                                    plt.yticks(fontsize = fs)

                                    plt.show()
                                if pdf_save==True:
                                    fig, axes = plt.subplots(figsize=(20, 8))
                                    axes.set_title("param {}\n ex {}, Neuron{}".format(self.date_i, ex, n))
                                    axes.plot(v)   #time[0:int(T/dt)-1],
                                    axes.set_xlabel('Time (sec)')
                                    axes.set_ylim(-110,40)
                                    axes.set_rasterized(True)
                                    pdf.savefig(fig)
                                    plt.close()
                                    del v 
                                    gc.collect()
                                # print("max_fre", maxfre)
                                # print("sp_num", sp_c)
                                # print(pattern)
                        else:
                            if n%3==0:
                                for tb in range(self.tbs):
                                    v_file = self.drc+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
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
                                    plt.plot(v)   #time[0:int(T/dt)-1],
                                    plt.ylim(-100, 40)
                                    plt.xlabel('Time (sec)',fontsize = fs)
                            #             plt.xlim(0,2)
                                        #plt.ylim(0,1)
                                    plt.xticks(fontsize = fs)
                                    plt.yticks(fontsize = fs)

                                    plt.show()
                                if pdf_save==True:
                                    fig, axes = plt.subplots(figsize=(20, 8))
                                    axes.set_title("param {}\n ex {}, Neuron{}".format(self.date_i, ex, n))
                                    axes.plot(v)   #time[0:int(T/dt)-1],
                                    axes.set_xlabel('Time (sec)')
                                    axes.set_ylim(-110,40)
                                    axes.set_rasterized(True)
                                    pdf.savefig(fig)
                                    plt.close()   
                                    del v 
                                    gc.collect()     
                    except:
                        traceback.print_exc()
        if pdf_save ==True:
            pdf.close()
            print("pdf_saving_end")
        
        
    def multi_cal(self, cores, exs, cv_w, moving, v_off, ISI_M_r, counts, T_w="None", stat = False, raster=False):
        args = []
        if T_w == "None":
            T = self.T
        else:
            T = T_w
        
        if stat ==True:
            # dt = 0.01
            p=os.getcwd()  #/net_search
            try:
                for i in range(counts):
                    cv_dis=np.load(p+"/CV_random_dis/ISIr_{}_{}/N{}_T{}_dt{}_3/CV_w_{}.npy".format(ISI_M_r[0],ISI_M_r[1], self.N, T, 0.01, i)).astype('float32')
                    if i==0:
                        cv_dis_all = cv_dis
                    else:
                        cv_dis_all = np.concatenate([cv_dis_all, cv_dis]).astype('float32')
                
                self.cv_dis_all = np.array(sorted(cv_dis_all))  # small-> large
            except:
                traceback.print_exc()
                self.cv_dis_all = np.zeros(1)
        for num, ex in enumerate(exs):
            args.append((num, ex, T, cv_w, moving, v_off, stat, raster))
        print(f'analyze networks: using {cores} cores')
        with Pool(processes=cores) as pool:
            pool.map(self.calc_values, args)
            
    
    def calc_values(self, args):
        num, ex, T, cv_w, moving, v_off, stat, raster = args
        self.df_np[num*14]=float(ex)
        
        if type(ex) != str:
            ex = round(ex,2)
            ex_in = round((ex + self.ex_diff), 2)
        else:
            ex_in = float(ex) +self.ex_diff
            ex_in = f"{ex_in:,.2f}"
            
        try:
            v_file = self.drc+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(0)+"_"+str(0)+".bin" 
            f2= open(v_file, "rb")
            rectype = np.dtype(np.float64)
            v_tb = np.fromfile(f2, dtype=rectype)
            dt = round(self.Tp/v_tb.shape[0],4)
        except:
            print("file_Error, ex{}".format(ex))
            traceback.print_exc()
            dt = self.dt
            
        flag=0
#             pattern_li = []
#             pattern_id=[]
        wake_c = 0
        sleep_c = 0
        wake_ex_c =0
        sleep_ex_c=0
        
        
        print("analyze exp", ex)
        
#             sp_c_li = []
        sp_fre_li = np.zeros(self.N)
        spts = np.zeros((self.N, int((T-v_off)/dt)), dtype = "int8")
        for n in range(self.N):
            try:
                for tb in range(self.tbs):
                    v_file = self.drc+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
                    f2= open(v_file, "rb")
                    rectype = np.dtype(np.float64)
                    v_tb = np.fromfile(f2, dtype=rectype).astype("float32")
                    if tb==0:
                        v=v_tb
                    else:
                        v=np.append(v, v_tb)
                    f2.close()

                pattern, maxfre, sp_c, peak_np = self.fre_spike(v[int(v_off/dt):int(T/dt)], int(T-v_off), dt)
                
                if np.any(np.isinf(v)) or np.any(np.isnan(v)):
                    flag = 1
                    print("nan_Error, can't make DF")
                    self.df_np[num*14+0]=1
#                     self.df_np[num][1:] = -1
                    for i in range(12):
#                         if i==0:
#                             continue
                        self.df_np[num*14 +i+2] = -1
                    break  #N
                
                if flag==1:
                    continue

                if pattern == "SWS" or pattern == "SWS_HIGH_FR":
                    sleep_c+=1
                    if n<self.NE:
                        sleep_ex_c+=1
                if pattern == "AWAKE" or pattern == "AWAKE_HIGH_FR":
                    wake_c+=1
                    if n<self.NE:
                        wake_ex_c+=1

#                     pattern_li.append(pattern)
                sp_fre_li[n]=(sp_c/(T-v_off)*1000)

                try:
                    # if maxfre!=0 and sp_c!=0:
                    #     if pattern!= "ERROR" and pattern != "EXCLUDED": 
                    #     # if pattern == "SWS" or pattern == "SWS_HIGH_FR" or pattern == "AWAKE" or pattern == "AWAKE_HIGH_FR" or pattern == "SWS_FEW_SPIKES" or pattern == "RES":
                    spts[n]= peak_np
        #                                                             print("{}, sp_add".format(n))
#                                     pattern_id.append(n)
                except:
                    print("file_Error, cant make DF")
                    traceback.print_exc()
                    flag = 1
                    self.df_np[num*14+1]=2
                    for i in range(12):
#                         if i==0:
#                             continue
                        self.df_np[num*14 +i+2] = -1
                    break  #N end
            except:
                print("file_Error, cant make DF")
                traceback.print_exc()
                flag=1
                self.df_np[num*14+1]=2
                for i in range(12):
#                         if i==0:
#                             continue
                        self.df_np[num*14 +i+2] = -1
                break  #N end
            
        if raster == True:
            plt.figure(figsize=(23, 15))
            Lt = int((T-v_off)/dt)
            x2=  np.arange(0, T-v_off, dt)
#                 spT = spts[spts.sum(axis=1) > 0., :]  #spikeあるものだけ選ぶ
            #color_set = ['r', 'b', 'k', 'orange', 'c']
            for i in range(spts.shape[0]):
                t_sp = x2[spts[i, :] > 0.5]   # spike times
                plt.plot(t_sp/1000, i*np.ones(len(t_sp)), '.',
                        ms=8, markeredgewidth=0.1, color ="black")
            plt.title("ex: {}".format(ex) ,fontsize = fs)
            plt.xlabel('Time (sec)',fontsize = fs)
            plt.ylabel('Neuron ID', fontsize = fs)
            #plt.xlim(0, 1)
            plt.xticks(fontsize = fs_l)
            plt.yticks(fontsize = fs_l)
            plt.show()
                
        if flag!=1 :
            m_sp_fre =np.mean(sp_fre_li)
            var_sp_fre =np.var(sp_fre_li)
            m_sp_fre_ex =np.mean(sp_fre_li[0:self.NE])
            var_sp_fre_ex =np.var(sp_fre_li[0:self.NE])
            sleep_per = sleep_c/(self.N)*100
            wake_per = wake_c/(self.N)*100
            sleep_per_ex = sleep_ex_c/(self.NE)*100
            wake_per_ex = wake_ex_c/(self.NE)*100

            self.df_np[num*14+1] = 0  #%nan
            self.df_np[num*14+2] = sleep_per
            self.df_np[num*14+3] = wake_per
            self.df_np[num*14+4] = m_sp_fre
            self.df_np[num*14+5] = var_sp_fre
            self.df_np[num*14+8] = sleep_per_ex
            self.df_np[num*14+9] = wake_per_ex
            self.df_np[num*14+10] = m_sp_fre_ex
            self.df_np[num*14+11] = var_sp_fre_ex
            # print("mean_fre_ex", m_sp_fre_ex)
            # print("var_sp_fre_ex", var_sp_fre_ex)
            # print("%sleep_ex", sleep_per_ex)
            # print("%wake_ex", wake_per_ex)

            sps_cv = self.calc_cv(self.N,spts, T-v_off, dt, cv_w,  moving)
#             print("CV_sp_count/w", sps_cv)
            self.df_np[num*14+6] = sps_cv
            # sps_cv_ex = self.calc_cv(self.NE,spts, T-v_off, dt, cv_w,  moving)
            # print("CV_sp_count_ex/w", sps_cv_ex)
            self.df_np[num*14+12] = -1#sps_cv_ex
            
            if stat ==True:
                p = self.get_cv_pv(sps_cv)
                # p_ex = self.get_cv_pv(sps_cv_ex)
                self.df_np[num*14+7] = p
                self.df_np[num*14+13] = -1#p_ex
            else:
                self.df_np[num*14+7] = -1
                self.df_np[num*14+13] = -1


    def get_cv_pv(self, cv):
        if cv >= np.max(self.cv_dis_all):
            pv = 0
        else:
            minx = np.argmin(np.abs(self.cv_dis_all-cv))

            # low_p = minx/len(self.cv_dis_all)*100
            up_p = (len(self.cv_dis_all)-minx)/len(self.cv_dis_all)*100
            # if low_p<up_p:
            #     pv=low_p
            #     # print(low_p)
            # else:
            pv=up_p
                # print(up_p)
            print("p_value", pv)
        return pv
        
    def make_df(self):
        df_np = np.array(self.df_np).reshape(len(self.exs), 14)
#         exs = np.array(self.exs).reshape(-1,1)
        print(df_np.shape)
#         print(exs)
#         df_np = np.concatenate([exs, self.df_np], axis=1)
        df_cc = pd.DataFrame(df_np, columns=["exp","%nan","%sleep","%wake","mean_spike_fre","var_spike_fre",  "cv", "p", "%sleep_ex","%wake_ex","mean_spike_fre_ex","var_spike_fre_ex", "cv_ex", "p_ex"])
        # df_cc.to_excel(self.drc + "cc.xlsx")
        df_cc.to_excel(self.savef + "_cc.xlsx")
        print(df_cc)
        nan_exs = df_cc[df_cc["%nan"]==1]["exp"]
        nan_exs = nan_exs.tolist()
        print(("nan_exp: {}\n".format(nan_exs)))
        error_exs = df_cc[df_cc["%nan"]==2]["exp"]
        error_exs = error_exs.tolist()
        print(("error_exp: {}\n".format(error_exs)))          

        return df_cc
            
            
    def calc_cv(self, N, spts, T, dt, cv_w,  moving):
        wc = int((T-cv_w)/moving+1)
        sp_li = []
        for m in range(wc):
            if m*moving/dt>0 and m*moving/dt + cv_w/dt  < (T)/dt:
                sp_sum = np.sum(spts[0:N,int(m*moving/dt):int(m*moving/dt + cv_w/dt)])
                sp_li.append(sp_sum)
        if np.mean(sp_li)!=0:
            sps_cv = np.std(sp_li)/np.mean(sp_li)
        else:
            sps_cv =0
        print("CV_sp_count/w", sps_cv)
        return sps_cv
    
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
       

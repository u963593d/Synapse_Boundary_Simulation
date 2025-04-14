# -*- coding: utf-8 -*-

""" 
This is the analysis module for Averaged Neuron (AN) model. In this module, 
you can analyze firing patterns from AN model, mainly using frequency and
spike analysis. 
"""

__author__ = 'Fumiya Tatsuki, Kensuke Yoshida, Tetsuya Yamada, \
              Takahiro Katsumata, Shoi Shi, Hiroki R. Ueda'
__status__ = 'Published'
__version__ = '1.0.0'
__date__ = '15 May 2020'


import os
# import sys
"""
LIMIT THE NUMBER OF THREADS!
change local env variables BEFORE importing numpy
"""
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from enum import Flag, auto
import numpy as np
from scipy.signal import periodogram, find_peaks
from scipy import signal
from typing import Optional
import matplotlib.pyplot as plt
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


class eval_wave:
    def __init__(self,model, i,NE,NI,con_M,con_M_in,con_ex_ex,con_ex_in,con_in_ex, con_in_in,g_nap_SD,gabac,T, Tp, dt,con_log, drc, fig_dir,exs, ex_diff):
        self.i = i
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
        self.g_nap_SD=g_nap_SD
        self.gabac=gabac
        self.T = T
        self.Tp=Tp
        self.dt = dt
        # self.con_log = con_log
        self.drc = drc
        # self.fig_dir = fig_dir
        self.x = np.arange(0, T, dt)/1000
        self.df_np = np.zeros((len(exs),13))
        
        
        if con_log == "True":
            self.savef = fig_dir+"{}_{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_gnap{}_gabac{}_T{}_dt{}".format(model, i, self.NE,self.NI,self.con_M,self.con_ex_ex,self.con_M_in, self.con_in_in,self.g_nap_SD,self.gabac,self.T,self.dt)
        else:
            self.savef = fig_dir+"{}_{}_NE{}_NI{}_conth{}_{}_{}_{}_gnap{}_gabac{}_T{}_dt{}".format(model, i, self.NE,self.NI,self.con_ex_ex,self.con_ex_in,self.con_in_ex, self.con_in_in,self.g_nap_SD,self.gabac,self.T,self.dt)
        

    def calc_values(self, exs, cv_w, moving, v_off, ISI_M_r, counts, T_w="None", calc =True, graph=True, pdf_save=False,stat = False):
        # self.exs = ex
        # s
        if T_w == "None":
            T = self.T
        else:
            T = T_w
            
        if stat ==True:
            dt = 0.01
            p=os.getcwd()  #/net_search/anmodel/
            try:
                for i in range(counts):
                    cv_dis=np.load(p+"/CV_random_dis/ISIr_{}_{}/N{}_T{}_dt{}/CV_w_{}.npy".format(ISI_M_r[0],ISI_M_r[1], self.N, T, dt, i)).astype('float32')
                    if i==0:
                        cv_dis_all = cv_dis
                    else:
                        cv_dis_all = np.concatenate([cv_dis_all, cv_dis]).astype('float32')
                
                self.cv_dis_all = np.array(sorted(cv_dis_all))  # small-> large
            except:
                traceback.print_exc()
                self.cv_dis_all = np.zeros(1)
        if pdf_save==True:
            
            pdf = PdfPages(self.savef+".pdf")
            
        for num, ex in enumerate(exs):
            if type(ex) != str:
                ex = round(ex,2)
                ex_in = round((ex + self.ex_diff), 2)
            else:
                ex_in = float(ex) +self.ex_diff
                ex_in = f"{ex_in:,.2f}"
            flag=0
#             pattern_li = []
#             pattern_id=[]
            wake_c = 0
            sleep_c = 0
            wake_ex_c =0
            sleep_ex_c=0
            
            # print("param_i",i)
            print("NE", self.NE)
            print("NI", self.NI)
            print("g_nap_SD", self.g_nap_SD)
            print("gaba_c", self.gabac)
           
            print("con_M", self.con_M)
            print("con_M_in", self.con_M_in)
            print("con_ex_ex", self.con_ex_ex)
            # print("con_ex_in", con_ex_in)
            # print("con_in_ex", con_in_ex)
            print("con_in_in", self.con_in_in)
            print("exp", ex)
            
#             sp_c_li = []
            sp_fre_li = np.zeros(self.N)
            spts = np.zeros((self.N, int((T-v_off)/self.dt)), dtype = "int8")
            for n in range(self.N):
                try:
                    for tb in range(self.tbs):
                        v_file = self.drc+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
                        f2= open(v_file, "rb")
                        rectype = np.dtype(np.float64)
                        v_tb = np.fromfile(f2, dtype=rectype)
                        if tb==0:
                            v=v_tb
                        else:
                            v=np.append(v, v_tb)
                        f2.close()
#                         if n<self.NE:
#                             if n==0:
#                                 vm_li = v
#                             else:
#                                 vm_li=np.append(vm_li, v)

                    pattern, maxfre, sp_c, peak_np = self.fre_spike(v[int(v_off/self.dt):int(T/self.dt)], int(T-v_off), self.dt)
                    if graph==True:
                        if self.N > 60 and self.N < 200:
                            if n%10 ==0:
                                plt.figure(figsize=(15, 4))
                                plt.title("Neuron{}".format(n))
                                plt.plot(self.x,v)   #time[0:int(T/dt)-1],
                                plt.ylim(-100, 40)
                                plt.xlabel('Time (sec)',fontsize = fs)
                        #             plt.xlim(0,2)
                                    #plt.ylim(0,1)
                                plt.xticks(fontsize = fs)
                                plt.yticks(fontsize = fs)

                                plt.show()
                                # print("max_fre", maxfre)
                                # print("sp_num", sp_c)
                                # print(pattern)
                        elif self.N > 200:
                            if n%25 ==0:
                                plt.figure(figsize=(15, 4))
                                plt.title("Neuron{}".format(n))
                                plt.plot(self.x,v)   #time[0:int(T/dt)-1],
                                plt.ylim(-100, 40)
                                plt.xlabel('Time (sec)',fontsize = fs)
                        #             plt.xlim(0,2)
                                    #plt.ylim(0,1)
                                plt.xticks(fontsize = fs)
                                plt.yticks(fontsize = fs)

                                plt.show()
                                # print("max_fre", maxfre)
                                # print("sp_num", sp_c)
                                # print(pattern)
                        else:
                            plt.figure(figsize=(15, 4))
                            plt.title("Neuron{}".format(n))
                            plt.plot(self.x,v)   #time[0:int(T/dt)-1],
                            plt.ylim(-100, 40)
                            plt.xlabel('Time (sec)',fontsize = fs)
                    #             plt.xlim(0,2)
                                #plt.ylim(0,1)
                            plt.xticks(fontsize = fs)
                            plt.yticks(fontsize = fs)

                            plt.show()
#                             print("max_fre", maxfre)
#                             print("sp_num", sp_c)
#                             print(pattern)

                    if pdf_save==True:
                        if self.NE+self.NI > 60 and self.NE+self.NI < 200:
                            if n%10 ==0:
                                fig, axes = plt.subplots(figsize=(20, 8))
                                axes.set_title("param {}\n ex {}, Neuron{}".format(self.i, ex, n))
                                axes.plot(self.x,v)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)')
                                axes.set_ylim(-110,40)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()
                                
                        elif self.NE+self.NI> 200:
                            if n%25 ==0:
                                fig, axes = plt.subplots(figsize=(20, 8))
                                axes.set_title("param {}\n ex {}, Neuron{}".format(self.i, ex, n))
                                axes.plot(self.x,v)   #time[0:int(T/dt)-1],
                                axes.set_xlabel('Time (sec)')
                                axes.set_ylim(-110,40)
                                axes.set_rasterized(True)
                                pdf.savefig(fig)
                                plt.close()
                        else:
                            fig, axes = plt.subplots(figsize=(20, 8))
                            axes.set_title("param {}\n ex {}, Neuron{}".format(self.i, ex, n))
                            axes.plot(self.x,v)   #time[0:int(T/dt)-1],
                            axes.set_xlabel('Time (sec)')
                            axes.set_ylim(-110,40)
                            axes.set_rasterized(True)
                            pdf.savefig(fig)
                            plt.close()

                    if np.any(np.isinf(v)) or np.any(np.isnan(v)):
                        flag = 1
                        print("nan_Error, can't make DF")
                        self.df_np[num][0]=1
                        self.df_np[num][1:] = -1
                        break  #N
                        # continue
            #                                                         pattern, maxfre, sp_c, peak_np = fre_spike(v, T, dt)
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
#                     sp_c_li.append(sp_c)
                    # print(pattern)
                    # print("spike_count: ", sp_c)
                    # print("maxfre: ", maxfre)

                    try:
                        if maxfre!=0 and sp_c!=0:
                            if pattern!= "ERROR" and pattern != "EXCLUDED": 
                            # if pattern == "SWS" or pattern == "SWS_HIGH_FR" or pattern == "AWAKE" or pattern == "AWAKE_HIGH_FR" or pattern == "SWS_FEW_SPIKES" or pattern == "RES":
                                spts[n]= peak_np
            #                                                             print("{}, sp_add".format(n))
#                                     pattern_id.append(n)
                    except:
                        print("file_Error, cant make DF")
                        traceback.print_exc()
                        flag = 1
                        self.df_np[num][0]=2
                        self.df_np[num][1:] = -1
                        break  #N end
                except:
                    print("file_Error, cant make DF")
                    traceback.print_exc()
                    flag=1
                    self.df_np[num][0]=2
                    self.df_np[num][1:] = -1
                    break  #N end
                    

            if flag!=1 and calc==True:
                 #tc  end   ex end

                m_sp_fre =np.mean(sp_fre_li)
                var_sp_fre =np.var(sp_fre_li)
                m_sp_fre_ex =np.mean(sp_fre_li[0:self.NE])
                var_sp_fre_ex =np.var(sp_fre_li[0:self.NE])
                sleep_per = sleep_c/(self.N)*100
                wake_per = wake_c/(self.N)*100
                sleep_per_ex = sleep_ex_c/(self.NE)*100
                wake_per_ex = wake_ex_c/(self.NE)*100


                self.df_np[num][0] = 0
                self.df_np[num][1] = sleep_per
                self.df_np[num][2] = wake_per
                self.df_np[num][3] = m_sp_fre
                self.df_np[num][4] = var_sp_fre
                self.df_np[num][7] = sleep_per_ex
                self.df_np[num][8] = wake_per_ex
                self.df_np[num][9] = m_sp_fre_ex
                self.df_np[num][10] = var_sp_fre_ex
                print("mean_fre_ex", m_sp_fre_ex)
                print("var_sp_fre_ex", var_sp_fre_ex)
                print("%sleep_ex", sleep_per_ex)
                print("%wake_ex", wake_per_ex)

                #                     wc = int((T-w)/moving+1)
                #                                                 sp_fre_nonzero = [item for item in sp_fre_li if item != 0]
                #                                                 fre_pro = math.prod(sp_fre_nonzero)
                #                                                 fre_pro_ex = math.prod(sp_fre_nonzero[0:NE])
                #                                                 CC_li = []
                #                                                 for m in range(NE+NI):
                #                                                     print("m",m)
                #                                                     sps_ref = spts[m]
                #                                                     CC_sum_n =0
                #                                                     for k, j in enumerate(sps_ref):
                #                                                         if j ==1:
                #                                                             if k-w/dt >=0 and k+w/dt < (T-v_off)/dt:
                #                                                                 CC_sum_n += np.sum(spts[:, int(k-w/dt):int(k+w/dt)])
                #                                                     CC_li.append(CC_sum_n/(((fre_pro)**(1/(NE+NI))))/(NE+NI))  # *np.mean(sp_fre_li)  
                #                                                 CC = np.mean(CC_li)
                #                                                 print("CC_mean", CC)
                #                                                 sps_cc_li.append(CC)

                #                                                 CC_ex_li = []
                #                                                 for m in range(NE):
                #                                                     print("m",m)
                #                                                     sps_ref = spts[m]
                #                                                     CC_sum_n =0
                #                                                     for k, j in enumerate(sps_ref):
                #                                                         if j ==1:
                #                                                             if k-w/dt >=0 and k+w/dt < (T-v_off)/dt:
                #                                                                 CC_sum_n += np.sum(spts[0:NE, int(k-w/dt):int(k+w/dt)])
                #                                                     CC_ex_li.append(CC_sum_n/(((fre_pro_ex)**(1/(NE))))/(NE))  # *np.mean(sp_fre_li)  
                #                                                 CC_ex = np.mean(CC_ex_li)
                #                                                 print("CC_mean_ex", CC_ex)
                #                                                 sps_cc_ex_li.append(CC_ex)

                sps_cv = self.calc_cv(self.N,spts, T-v_off, cv_w,  moving)
    #             print("CV_sp_count/w", sps_cv)
                self.df_np[num][5] = sps_cv
                sps_cv_ex = self.calc_cv(self.NE,spts, T-v_off, cv_w,  moving)
                print("CV_sp_count_ex/w", sps_cv_ex)
                self.df_np[num][11] = sps_cv_ex
                
                if stat ==True:
                    p = self.get_cv_pv(sps_cv)
                    p_ex = self.get_cv_pv(sps_cv_ex)
                    self.df_np[num][6] = p
                    self.df_np[num][12] = p_ex
                else:
                    self.df_np[num][6] = -1
                    self.df_np[num][12] = -1
                    
                
        if pdf_save==True:
            pdf.close()

    def get_cv_pv(self, cv):
        if cv >= np.max(self.cv_dis_all):
            pv = 0
        else:
            minx = np.argmin(np.abs(self.cv_dis_all-cv))

            low_p = minx/len(self.cv_dis_all)*100
            up_p = (len(self.cv_dis_all)-minx)/len(self.cv_dis_all)*100
            if low_p<up_p:
                pv=low_p
                # print(low_p)
            else:
                pv=up_p
                # print(up_p)
            print("p_value", pv)
        return pv
        
    def make_df(self):
        exs = np.array(self.exs).reshape(-1,1)
        df_np = np.concatenate([exs, self.df_np], axis=1)
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
            
            
    def calc_cv(self, N, spts, T, cv_w,  moving):
        wc = int((T-cv_w)/moving+1)
        sp_li = []
        for m in range(wc):
            if m*moving/self.dt>0 and m*moving/self.dt + cv_w/self.dt  < (T)/self.dt:
                sp_sum = np.sum(spts[0:N,int(m*moving/self.dt):int(m*moving/self.dt + cv_w/self.dt)])
                sp_li.append(sp_sum)
        if np.mean(sp_li)!=0:
            sps_cv = np.var(sp_li)/np.mean(sp_li)
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
        
def fre_spike_i(v, T, dt):
    
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
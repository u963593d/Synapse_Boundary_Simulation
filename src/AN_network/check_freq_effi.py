
import os
import sys
import traceback
from scipy.signal import find_peaks
from scipy.signal import periodogram
from scipy import signal
import numpy as np
import gc

from statannot import add_stat_annotation

mode="w"
op = "_net_effi"
op_param = "_noinin"
op_cr = "_net_crw"
model_op=op

model_pre = "AN" 
cpre_r = 1.0
cpost_r =1.0


rho_ini = 0.5
smin_ampa = 0.5
smax_ampa=1.5
f_pair ="ex_pair.csv"
f_end = "sw_end.csv"

T = 60000  #ms
Tp = 10000  #ms

dt = 0.05
v_off = 0

class eval_wave_lr:
    def __init__(self, NE, NI, T, Tp, dt, v_off, drc_neu, drc_syn, tbs):
        self.N = NE + NI
        self.NE = NE
        self.NI = NI
     
        self.tbs = int(T/Tp)# tbs #100 # int(T/Tp)
        self.T = T
        self.Tp=Tp
        self.dt = dt
        self.drc_neu = drc_neu
        self.drc_syn = drc_syn

        self.x= np.arange(0,tbs*Tp,dt)/1000
    
    def v_of_exp(self, n, ex):
        ex_in = ex+self.ex_diff
        for tb in range(self.tbs):
            v_file = self.drc_neu+ "ex"+str(ex)+"_"+"ex_in"+str(ex_in)+"_"+"N"+str(n)+"_"+str(tb)+".bin" 
            f2= open(v_file, "rb")
            rectype = np.dtype(np.float64)
            v_tb = np.fromfile(f2, dtype=rectype)
            if tb==0:
                v=v_tb
            else:
                v=np.append(v, v_tb)
            f2.close()
        return v
    
    def rho_of_exp(self, pre_syn, n):
        
        for tb in range(self.tbs):
            rho_file = self.drc_syn+"rho"+str(pre_syn)+"-"+str(n)+"_"+str(tb)+".bin"
            frho= open(rho_file, "rb")
            rectype = np.dtype(np.float64)
            rho_tb = np.fromfile(frho, dtype=rectype)
            if tb==0:
                rho=rho_tb
            else:
                rho=np.append(rho, rho_tb)
            frho.close()
        return rho
    
    def ave_rho(self, N,  start):
        try:
            v_file = self.drc_neu +"N"+str(0)+"_"+str(0)+".bin" 
            f2= open(v_file, "rb")
            rectype = np.dtype(np.float64)
            v_tb = np.fromfile(f2, dtype=rectype)
            dt = round(self.Tp/v_tb.shape[0],4)
            print("dt_calc", dt)
#             print("ex {}, dt{}".format(ex, dt))
#             self.x = np.arange(0, self.T, dt)/1000
        except:
            print("dt_calc_Error")
            traceback.print_exc()
        rho_m_li = np.empty(0)
        for n in range(N):
            for s in range(N):
                cpre_file = self.drc_syn+"cpre"+str(s)+"-"+str(n)+"_"+str(0)+".bin"
                if os.path.exists(cpre_file):
                    rho = self.rho_of_exp(s, n)
                    rho_m = np.mean(rho[int(start/dt):])
                    rho_m_li = np.append(rho_m_li, rho_m)
        return np.mean(rho_m_li)
    
    
    def ave_rho2(self, N,  start):
        try:
            v_file = self.drc_neu +"N"+str(0)+"_"+str(0)+".bin" 
            f2= open(v_file, "rb")
            rectype = np.dtype(np.float64)
            v_tb = np.fromfile(f2, dtype=rectype)
            dt = round(self.Tp/v_tb.shape[0],4)
            print("dt_calc", dt)
#             print("ex {}, dt{}".format(ex, dt))
#             self.x = np.arange(0, self.T, dt)/1000
        except:
            print("dt_calc_Error")
            traceback.print_exc()
        syN=0
        for n in range(N):
            for s in range(N):
                cpre_file = self.drc_syn+"cpre"+str(s)+"-"+str(n)+"_"+str(0)+".bin"
                if os.path.exists(cpre_file):
                    syN+=1
                    
        rho_m_li = np.zeros((syN, int((self.T-start)/dt)))
        c=0
        for n in range(N):
            for s in range(N):
                cpre_file = self.drc_syn+"cpre"+str(s)+"-"+str(n)+"_"+str(0)+".bin"
                if os.path.exists(cpre_file):
                    rho = self.rho_of_exp(s, n)
                    rho_m_li[c]=rho[int(start/dt):]
                    c+=1
#                     rho_m = np.mean(rho[int(start/dt):])
#                     rho_m_li = np.append(rho_m_li, rho_m)
        return rho_m_li
    
    
    
    
    def mean_fre(self, N, start):

        spN=np.zeros(N, dtype="int64")
        flag = 0
        for n in range(N):
            for tb in range(self.tbs):
                v_file = self.drc_neu +"N"+str(n)+"_"+str(tb)+".bin" 
                f2= open(v_file, "rb")
                rectype = np.dtype(np.float64)
                v_tb = np.fromfile(f2, dtype=rectype)
                if tb==0:
                    v=v_tb
                else:
                    v=np.append(v, v_tb)
                f2.close()
            pattern, maxfre, sp_c, peak_np = self.fre_spike(v, start)
            spN[n]=sp_c/((self.T-start)/1000)
            if sp_c==0:
                flag=1
                break
        m_fre= np.mean(spN)
#         print("mean spike fre", m_fre)
        return m_fre, flag
            
    
    def fre_spike(self, v, start):
#         v=v[0:int(T/dt)]
        try:
            dt = round(self.T/v.shape[0],4)

        except:
            print("dt_calc_Error")
            traceback.print_exc()
        # print("dt", dt)
        T = int(v.shape[0]*dt-start) #ms
        # print("v.shape[0]",v.shape[0])
        # print("T", T)
        
        if np.any(np.isinf(v)) or np.any(np.isnan(v)):
            print("fre_spike: nan")
            return "Exclude", 0, 0, np.zeros(int(T/dt), dtype="int8")
        else:
#             wavech = anmodel.analysis.WaveCheck(1000)
#             pattern = wavech.pattern(v, T, dt, pr=False)
            detv: np.ndarray = signal.detrend(v)
            max_potential: float = max(detv)
            f, spw = periodogram(detv, fs=1/dt*1000)
            maxamp: float = max(spw)
            nummax: int = spw.tolist().index(maxamp)
            maxfre: float = f[nummax]

            peaks = find_peaks(v[int(start/dt):], height=-20)[0]
#             print(v.shape)
            peak_np = np.zeros(int(T/dt), dtype="uint8")
            peak_np[peaks]=1
#             print(peak_np.shape)
            return "sleep", maxfre, len(peaks), peak_np
            

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

con_M = float(args[c])  #[0.5, 1.0, 1.2, 1.5,1.7]   1.5, 2.1, 3.0 (N increase)
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

T2=int(args[c])
c+=1
Tp=int(args[c])
c+=1


tbs= int(T/Tp)
con_log = True


seed = int(args[c])
c+=1

init =(args[c])
c+=1


ex_w = float(args[c])
c+=1
ex_s = float(args[c])
c+=1

bifur = args[c]
c+=1
print("T,", T)
print("sw_fre_check")
print("ex_w", ex_w)

NI = int(NE*Ip/(100-Ip))

ex_diff=0

blockdim_x=NE+NI
start_m=10000  #ms
x = np.arange(0, T, dt)/1000  #sec

model_name = model_pre + model_op + bifur + "_"+lr + "_w"
model_name_s = model_pre + model_op + bifur + "_"+lr + "_s"

   
nowdir = os.getcwd()+"/"


n=0

drc ="./con_cu_i/{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/sminampa{}_smaxampa{}/{}_{}/".format(model_name,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in,smin_ampa,smax_ampa,lr,lr_p)
drc_s ="./con_cu_i/{}/NE{}_NI{}/conM{}_conSD{}_conMin{}_conSDin{}/sminampa{}_smaxampa{}/{}_{}/".format(model_name_s,NE, NI,con_M, con_ex_ex,con_M_in, con_in_in,smin_ampa,smax_ampa,lr,lr_p)




ex_in_w = ex_w #-2.0
ex_in_s = ex_s



savedir =  drc + date_i + "/"
savedir_s =  drc_s + date_i +"/"
savef=savedir + "wake.csv"
savef_s=savedir_s + "sleep.csv"

savef_cv=savedir + "wake_cv.csv"
savef_cv_s=savedir_s + "sleep_cv.csv"

fref=savedir + "m_wake_fre.csv"
fref_s=savedir_s + "m_sleep_fre.csv"

fref_dir =  drc + date_i + "/fre_exs/"
fref_ex = fref_dir + "{}.csv".format(round(float(ex_w),2))
fref_dir_s =  drc_s + date_i + "/fre_exs/"
fref_ex_s = fref_dir_s + "{}.csv".format(round(float(ex_s),2))

try:
    os.makedirs(fref_dir)
except FileExistsError:
    print("{} is already exist".format(fref_dir)) 
try:
    os.makedirs(fref_dir_s)
except FileExistsError:
    print("{} is already exist".format(fref_dir_s)) 


# drc_i = drc + date_i + "/exs/{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/".format(ex_w, seed,init,  T, Tp,dt)
# drc_i_s = drc_s + date_i + "/exs/{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/".format(ex_s,seed,init,  T, Tp,dt)
# if not os.path.exists(drc_i):
drc_i = drc + date_i + "/exs/{}/seed_{}/init_{}/T_{}/Tp_{}/".format(ex_w,seed,init,  T, Tp)
drc_i_s = drc_s + date_i + "/exs/{}/seed_{}/init_{}/T_{}/Tp_{}/".format(ex_s,seed,init,  T, Tp)

drc_neu = drc_i + "neu/"
drc_syn = drc_i + "syn/"
drc_neu_s = drc_i_s + "neu/"
drc_syn_s = drc_i_s + "syn/"

try:
    wave_check = eval_wave_lr(NE, NI, T, Tp, dt,  v_off, drc_neu, drc_syn, tbs)
    wave_check_s = eval_wave_lr(NE, NI, T, Tp, dt,  v_off, drc_neu_s, drc_syn_s, tbs)

    if not os.path.exists(fref_ex):
        fre_m, flag = wave_check.mean_fre(NE+NI, start_m)
        if flag==1:
            fre_m=-1
            fre_m_v=np.array([-1])
            print("{} wake including nan -> -1".format(date_i))
        else:
            fre_m_v = np.array([fre_m])
        print("{}, ex_w{}   fre_m_w {}".format(date_i,ex_w, fre_m_v))
        np.savetxt(fref_ex, fre_m_v, delimiter=',')
   

    if not os.path.exists(fref_ex_s):
        fre_m_s, flag_s = wave_check_s.mean_fre(NE+NI,start_m)
        if flag_s==1:
            fre_m_s=-1
            fre_m_s_v=np.array([-1])
            print("{} sleep including nan -> -1".format(date_i))
        else:
            fre_m_s_v = np.array([fre_m_s])
        print("{}, ex_s{},  fre_m_s {}".format(date_i,ex_s, fre_m_s_v))
        np.savetxt(fref_ex_s, fre_m_s_v, delimiter=',')
except:
    traceback.print_exc()


# combination of freq in fre files

fre_w_li = []
if os.path.exists(drc + date_i+"/exs/"):
    exs = os.listdir(drc + date_i+"/exs/")
    
    #wake
    
    for ex in exs:
        fref_ex = drc + date_i + "/fre_exs/{}.csv".format(round(float(ex),2))
        if os.path.exists(fref_ex):
            fre_w = np.genfromtxt(fref_ex, delimiter=',', dtype=np.float64)
        
            fre_w_li.append(fre_w.tolist())
    print("fre_w_li", fre_w_li)

fre_s_li = []
if os.path.exists(drc_s + date_i+"/exs/"):
    exs_s = os.listdir(drc_s + date_i+"/exs/")
        
    #sleep
    
    for ex in exs_s:
        fref_ex_s = drc_s + date_i + "/fre_exs/{}.csv".format(round(float(ex),2))
        if os.path.exists(fref_ex_s):
            fre_s = np.genfromtxt(fref_ex_s, delimiter=',', dtype=np.float64)
            fre_s_li.append(fre_s.tolist())
    print("fre_s_li", fre_s_li)
    
fre_pairs=[]
fre_diffs = []
ex_pairs =[]

if len(fre_w_li) !=0 and len(fre_s_li) !=0:
    for m, fre_w in enumerate(fre_w_li):
    #     print("fre_w", fre_w)
        for n, fre_s in enumerate(fre_s_li):
            ex_w = float(exs[m])
            ex_s = float(exs_s[n])
    #         print("fre_s", fre_s)
            if fre_w!=-1 and fre_s!=-1 and np.abs(fre_w-fre_s) <=2 and ex_w < ex_s:
                fre_pairs.append([fre_w, fre_s])
                fre_diffs.append(np.abs(fre_w-fre_s))
                ex_pairs.append([ex_w, ex_s])


    if len(ex_pairs) !=0:
        fre_diff_min_in = np.argmin(np.array(fre_diffs))
        ex_w = ex_pairs[fre_diff_min_in][0]
        ex_s = ex_pairs[fre_diff_min_in][1]
        fre_w = fre_pairs[fre_diff_min_in][0]
        fre_s = fre_pairs[fre_diff_min_in][1]

        print("ex_w", ex_w)
        print("ex_s", ex_s)

        # drc_i = drc + date_i + "/exs/{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/".format(ex_w, seed,init,  T, Tp,dt)
        # drc_i_s = drc_s + date_i + "/exs/{}/seed_{}/init_{}/T_{}/Tp_{}/dt_{}/".format(ex_s,seed,init,  T, Tp,dt)
        # if not os.path.exists(drc_i):
        drc_i = drc + date_i + "/exs/{}/seed_{}/init_{}/T_{}/Tp_{}/".format(ex_w,seed,init,  T, Tp)
        drc_i_s = drc_s + date_i + "/exs/{}/seed_{}/init_{}/T_{}/Tp_{}/".format(ex_s,seed,init,  T, Tp)

        drc_neu = drc_i + "neu/"
        drc_syn = drc_i + "syn/"
        drc_neu_s = drc_i_s + "neu/"
        drc_syn_s = drc_i_s + "syn/"              

        try:
            wave_check = eval_wave_lr(NE, NI, T, Tp, dt,  v_off, drc_neu, drc_syn, tbs)
            wave_check_s = eval_wave_lr(NE, NI, T, Tp, dt,  v_off, drc_neu_s, drc_syn_s, tbs)


            if not os.path.exists(drc+date_i+"/"+f_end):
                np.savetxt(drc+date_i+"/"+f_end, np.array([ex_w, ex_s]), delimiter=",")
                print("save_sw_end ", date_i)

           #fre
            if not os.path.exists(fref):
                print("{}, fre_w {}".format(date_i, fre_w))
                np.savetxt(fref, np.array([fre_w]), delimiter=',')


            if not os.path.exists(fref_s):
                print("{}, fre_s {}".format(date_i, fre_s))
                np.savetxt(fref_s, np.array([fre_s]), delimiter=',')

            #mean effi  cv  
            if not os.path.exists(savef):
                rho_li_w = wave_check.ave_rho2(NE, start_m)
    #             

                effis = np.array([n, np.mean(rho_li_w)])
                print("effis_wake", effis)
                np.savetxt(savef, effis, delimiter=',')
    #             np.savetxt(savef, np.mean(rho_li_w, axis=0), delimiter=',')  #time change

                rho_li_w=np.std(rho_li_w, axis=0)/np.mean(rho_li_w, axis=0)
                ave_cv = np.array([n, np.mean(rho_li_w)])
                print("cv_wake", ave_cv)

                np.savetxt(savef_cv, ave_cv, delimiter=',')
    #             np.savetxt(savef_cv, rho_li_w, delimiter=',')  #time change
                del rho_li_w
                gc.collect()

            if not os.path.exists(savef_s):
                rho_li_s = wave_check_s.ave_rho2(NE, start_m)
    #             print("{}, rho_m_s {}".format(date_i, rho_m_s))


                effis_s = np.array([n, np.mean(rho_li_s)])
                print("effis_sleep", effis_s)
                np.savetxt(savef_s, effis_s, delimiter=',')
    #             np.savetxt(savef_s, np.mean(rho_li_s, axis=0), delimiter=',')  #time change

                rho_li_s=np.std(rho_li_s, axis=0)/np.mean(rho_li_s, axis=0)
                ave_cv_s = np.array([n, np.mean(rho_li_s)])
                print("cv_sleep", ave_cv_s)
                np.savetxt(savef_cv_s, ave_cv_s, delimiter=',')
    #             np.savetxt(savef_cv_s, rho_li_s, delimiter=',')  #time change
                del rho_li_s
                gc.collect()




        except:
            traceback.print_exc()
    else:
        print("no ex_pairs")
else:
    print("no fre_w_li or fre_s_li")
    
    
           
    
    
    

import pickle
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('../')
sys.path.append('../anmodel')
import anmodel
import subprocess

#args
#search_bifurcation_single.py nav kva kvsi kir nmdar gabar doing_parameter_search doing_bifurcation_analysis collecting_bifurcated_parameters selecting_numbers bifurcated_molecule


#command
# python3 search_bifurcation_single.py 1 1 1 1 1 1 1 1 1 1 nmdar (cav)


# parameters
fs = 25  #fontsize
fs_l = 20  #axis_label fontsize

cores = 2  # cores using when parameter search for channel or receptor conductance
ncore = 2 # cores using when bifurcation analysis
time_h = 6  #duration of time for parameter search #hour

T = 5000                #analyzed Time for parameter search and bifurcation analysis
offset = 1000           #offset Time for parameter search and bifurcation analysis
dt = 0.1
Lt = int(T/dt)
step = int(10**(2))         #number of total steps 

param_range = np.linspace(-2, 2, step)   #range of the coefficients (original X 10^exs)


exs = np.arange(-2, 2.0, 0.25).tolist()    #coefficient range for output waveform plot
T_c = 3000  # ms                  #Time for output waveform plot
offset_c = 0                        #offset for output waveform plot

bifur_direction = "desyn->syn"

# -----------------------------------------


args = sys.argv

nav = int(args[1])
kva = int(args[2])
kvsi = int(args[3])
kir = int(args[4])
nmdar = int(args[5])
gabar = int(args[6])
param_search = int(args[7])  
bifur_analysis = int(args[8]) 
orders = []
collect_bifur_params = int(args[9]) 
select_numbers = int(args[10]) 
g_str = "g_"+args[11]

if bifur_direction == "desyn->syn":
    param_range = param_range[::-1]

if nav == 1 and kva == 1 and kvsi == 1 and kir == 1 and gabar == 1:  # NaK1
    keys = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi', 'g_cav',
            'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_nmdar', 'g_gabar', 't_ca']
    ext = "AN"
elif nav == 0 and kva == 1 and kvsi == 1 and kir == 1 and gabar == 1:  # K1
    keys = ['g_leak', 'g_kvhh', 'g_kva', 'g_kvsi', 'g_cav',
            'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K1"
elif nav == 1 and kva == 1 and kvsi == 1 and kir == 0 and gabar == 1:  # NaK3
    keys = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_kvsi',
            'g_cav', 'g_kca', 'g_nap',  'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK3"
elif nav == 0 and kva == 1 and kvsi == 1 and kir == 0 and gabar == 1:  # K3
    keys = ['g_leak',  'g_kvhh', 'g_kva', 'g_kvsi', 'g_cav',
            'g_kca', 'g_nap', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K3"
elif nav == 1 and kva == 0 and kvsi == 1 and kir == 0 and gabar == 1:  # NaK6
    keys = ['g_leak', 'g_nav', 'g_kvhh',  'g_kvsi', 'g_cav',
            'g_kca', 'g_nap',  'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK6"
elif nav == 0 and kva == 0 and kvsi == 1 and kir == 0 and gabar == 1:  # K6
    keys = ['g_leak',  'g_kvhh',  'g_kvsi', 'g_cav',
            'g_kca', 'g_nap', 'g_ampar', 'g_gabar', 't_ca']
elif nav == 1 and kva == 1 and kvsi == 0 and kir == 0 and gabar == 1:  # NaK5
    keys = ['g_leak', 'g_nav', 'g_kvhh', 'g_kva', 'g_cav',
            'g_kca', 'g_nap', 'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK5"
elif nav == 0 and kva == 1 and kvsi == 0 and kir == 0 and gabar == 1:  # K5
    keys = ['g_leak',  'g_kvhh', 'g_kva', 'g_cav',
            'g_kca', 'g_nap', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K5"
elif nav == 1 and kva == 0 and kvsi == 0 and kir == 1 and gabar == 1:  # NaK7
    keys = ['g_leak', 'g_nav', 'g_kvhh', 'g_cav', 'g_kca',
            'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK7"
elif nav == 0 and kva == 0 and kvsi == 0 and kir == 1 and gabar == 1:  # K7
    keys = ['g_leak', 'g_kvhh', 'g_cav', 'g_kca',
            'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K7"
elif nav == 1 and kva == 0 and kvsi == 1 and kir == 1 and gabar == 1:  # NaK4
    keys = ['g_leak', 'g_nav', 'g_kvhh',  'g_kvsi', 'g_cav',
            'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK4"
elif nav == 0 and kva == 0 and kvsi == 1 and kir == 1 and gabar == 1:  # K4
    keys = ['g_leak',  'g_kvhh',  'g_kvsi', 'g_cav', 'g_kca',
            'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K4"
elif nav == 1 and kva == 1 and kvsi == 0 and kir == 1 and gabar == 1:  # NaK2
    keys = ['g_leak', 'g_nav', 'g_kvhh',  'g_kva', 'g_cav',
            'g_kca', 'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "NaK2"
elif nav == 0 and kva == 1 and kvsi == 0 and kir == 1 and gabar == 1:  # K2
    keys = ['g_leak', 'g_kvhh',  'g_kva', 'g_cav', 'g_kca',
            'g_nap', 'g_kir', 'g_ampar', 'g_gabar', 't_ca']
    ext = "K2"
elif nav == 0 and kva == 0 and kvsi == 0 and kir == 0 and gabar == 1:  # K2
    keys = ['g_leak', 'g_kvhh',  'g_cav', 'g_kca',
            'g_nap', 'g_ampar', 'g_nmdar', 'g_gabar', 't_ca']
    ext = "K0"


model_pre = ext

avecon = 1
dup = 0  

sanA_bool = [
    1,  # leak channel
    nav,  # voltage-gated sodium channel
    1,  # HH-type delayed rectifier potassium channel
    kva,  # fast A-type potassium channel
    kvsi,  # slowly inactivating potassium channel
    1,  # voltage-gated calcium channel
    1,  # calcium-dependent potassium channel
    1,  # persistent sodium channel
    kir,  # inwardly rectifier potassium channel
    1,  # AMPA receptor
    nmdar,  # NMDA receptor
    gabar,  # GABA receptor
    1,  # calcium pu
]

san_x = anmodel.models.Xmodel(channel_bool=sanA_bool)

savedir = os.getcwd() + "/results/SWS_params/"
now = datetime.now()
date: str = f'{now.year}_{now.month}_{now.day}'

model_name =ext 

nowdir = os.getcwd() + "/"
os.makedirs(nowdir + "results_bifurcation/" , exist_ok=True)
dir_li = os.listdir(nowdir + "results_bifurcation/")
print(dir_li)
dup = 0
while (1):
    model_fol = date + "_" + str(dup) + "_" + model_name
    dir_flag = 0
    for dir_str in (dir_li):

        if dir_str == model_fol:
            dir_flag = 1
            break
    print("flag", dir_flag)
    if dir_flag == 0:
        break
    else:
        dup += 1

print(model_fol)
date = date + "_" + str(dup)

model_fol = date + "_" + model_name
os.makedirs(nowdir + "results_bifurcation/" + model_fol, exist_ok=True)

    

path = "SWS_"+date  # 
time = np.arange(0, T_c+offset, dt)/1000  # s
root_dir = os.getcwd() + "/"  # "/mnt/SSD3/averaged_neuron_modelDB/ipynb"

ave_cons = [avecon]

bifur_fol = root_dir + "/results_bifurcation/"+model_fol  + "/"+model_fol +"_"+g_str

bifur_total_data = bifur_fol+"/"+"bifurcation_Total.xlsx"
print(bifur_total_data)
fname_r = "/results/SWS_params"  # random_search
search_path = root_dir + "results/SWS_params/" +model_fol +"/" + "SWS_"+date  # random_search



def bifur_total(T, dt, offset, savedir, date, fname, path, cores, model_name, g_str, ave_con, step, param_range):
    p = os.getcwd()

    res_p = p + "/results_bifurcation/"+date+"_" + \
        model_name+"/"+date + "_" + model_name + "_"+g_str
    try:
        os.makedirs(res_p)
    except FileExistsError:
        print("file exist")
        pass

    save_ps = []
    bifur_cs = []
    bifur_chs = []
    c = 0
    for n in range(cores):
        #         if n !=1:
        #             continue
        fpath = savedir + date+"_"+fname + "/"+path + "_"+str(n) + ".pickle"
        print("file_path", fpath)
        with open(fpath, "rb") as f:
            iters = pickle.load(f)
            params = pickle.load(f)

        for i in range(len(params)):
            #             print("core{}, param{}".format(n,i))
            c += 1
#             if i!=4:
#                 continue
            pd_param = params.iloc[i, 0:]
            param_set = pd_param.to_dict()
            # print("param_set", param_set)

            save_p, bifur_c, bifur_ch = bifur_ana(
                T, dt, offset, param_set, res_p, g_str, ave_con, step, param_range, n, i)

            if bifur_c > 0 or bifur_ch > 0:
                bifur_cs.append(bifur_c)  # WAKE +
                bifur_chs.append(bifur_ch)  # WAKE +
                save_ps.append(save_p)

            if bifur_c > 0:
                print("core{}-param{}, Bifurcation!, AWAKE".format(n, i))

            if bifur_ch > 0:
                print("core{}-param{}, Bifurcation!, AWAKE_HIGH_FR".format(n, i))

            if (len(bifur_cs) + len(bifur_chs)) % 2 == 0 and (len(bifur_cs) + len(bifur_chs)) != 0:

                print("bifurcation_analysis core{} end".format(n))
                pd_bifur = pd.DataFrame(list(zip(save_ps, bifur_cs, bifur_chs)), columns=[
                                        "file", "wake_count", "wake_high_fr_count"])
                # syuukei file save
                pd_bifur.to_excel(res_p + "/"+"bifurcation_Total.xlsx")

    print("bifurcation_analysis end")
    pd_bifur = pd.DataFrame(list(zip(save_ps, bifur_cs, bifur_chs)), columns=[
                            "file", "wake_count", "wake_high_fr_count"])

    # syuukei file save
    pd_bifur.to_excel(res_p + "/"+"bifurcation_Total.xlsx")
    per_bifur = len(bifur_cs)/c * 100
    per_bifurh = len(bifur_chs)/c * 100
    print("total_params", c)
    print("num of bifurcated by wake", len(bifur_cs))
    print("bifurcation to wake%", per_bifur)
    print("num of bifurcated by wake with high FR", len(bifur_chs))
    print("bifurcation to wake with high FR%", per_bifurh)
    return pd_bifur

#bifurcation_analysis
def bifur_ana(T, dt, offset, param_set, res_p, g_str, ave_con, step, param_range, core_n, param_n):
    param_set_c = param_set.copy()
    g_ori = param_set_c[g_str]
    san_x = anmodel.models.Xmodel(channel_bool=sanA_bool)
    san_x.set_params(param_set)
#     down = Down_Judge_Null(10000, 10, 49999, 99999)

    bifur_c = 0
    bifur_ch = 0
    sws_c = 0
    sws_ch = 0
    pre_flag = 0
    s_flag = 0
    v_flag = 0
    ex_flag = 0
    s_vmin = 60

    try:
        #         s_ori  = san_x.Solvers_sanA(dt, T+offset)
        s_ori, info = san_x.run_odeint(int(1/dt*1000), int((T+offset)/1000))
        gate_a_ori = s_ori[int(offset/dt):, 2]
        v_ori: np.ndarray = s_ori[int(offset/dt):, 0]
        ori_ampai_max = np.max(param_set[g_str]*gate_a_ori*(v_ori-0))
        ori_v_min = np.min(v_ori)

        save_p = res_p + "/"+"core"+str(core_n)+"_"+"param"+str(param_n)
        g_li = np.zeros((2, len(param_range)))

        break_c = 0

        for n, param in enumerate(param_range):
            #             print("ex", param)
            #             param = 0.69
            # st = time()
            #         md = time()  #s

            #         if md-st > 10*60: #second
            #             break_c += 1
            #             break
            # print(n)
            g = g_ori*10**(param)*ave_con  # *original value
           # niter+=1
            param_set[g_str] = g

           # print(param_set)
            san_x.set_params(param_set)

            g_li[0][n] = g
            s: np.ndarray

            try:
                #                 s  = san_x.Solvers_sanA(dt, T+offset)
                s, info = san_x.run_odeint(
                    int(1/dt*1000), int((T+offset)/1000))
                gate_a = s[int(offset/dt):, 2]  # s_ampar
                v: np.ndarray = s[int(offset/dt):, 0]

#                 plt.figure(figsize=(23, 5))
#                 plt.plot(s[:,0])
#                 plt.xlabel('Time',fontsize = fs)
#                 plt.ylabel("V", fontsize = fs)
#                 plt.xticks(fontsize = fs_l)
#                 plt.yticks(fontsize = fs_l)
#                 plt.show()

                ampai_max = np.max(g*gate_a*(v-0))
                # , ori_ampai_max, ampai_max, ori_v_min)
                wave_check = anmodel.analysis.WaveCheck(1000)
                pattern: anmodel.analysis.WavePattern = wave_check.pattern(
                    v, T, dt, pr=False)
#
#
                v_min = np.min(v)

                if pattern.name == "SWS":
                    g_li[1][n] = 0
                    sws_c += 1
                    pre_flag = 1
                    s_vmin = v_min
                elif pattern.name == "SWS_HIGH_FR":
                    g_li[1][n] = 1
                    sws_ch += 1
                    pre_flag = -1
                    # s_vmin=v_min
                elif pattern.name == "SWS_FEW_SPIKES":
                    #                     if down.down_judge(param_set, sanbool)==1:
                    g_li[1][n] = 2
                    pre_flag = -1
#                     else:
#                         pattern.name == "AWAKE"
#                         g_li[1][n] = 3
                elif pattern.name == "AWAKE":
                    #                     if down.down_judge(param_set, sanbool)==1:
                    #                         pattern.name == "SWS_FEW_SPIKES"
                    #                         g_li[1][n] = 2
                    #                     else:
                    g_li[1][n] = 3
                    # print('Bifurcation!')

                    bifur_c += 1
                    if pre_flag == 1:
                        s_flag = 2
                        if s_vmin + 5 < v_min:
                            v_flag = 1

                    elif pre_flag == 3 and s_flag != 2:
                        s_flag = 4

                    pre_flag = -1

                elif pattern.name == "AWAKE_HIGH_FR":
                    #                     if down.down_judge(param_set, sanbool)==1:
                    #                         pattern.name == "SWS_FEW_SPIKES"
                    #                         g_li[1][n] = 2
                    #                     else:
                    g_li[1][n] = 4
                    pre_flag = -1
                    # bifur_ch += 1
                    # if pre_flag == 1:
                    #     s_flag = 2
                    #     if s_vmin + 5 < v_min:
                    #         v_flag=1
                    # elif pre_flag == 3 and s_flag!=2:
                    #     s_flag = 6
                    # pre_flag=-1
                elif pattern.name == "cyclic_firing_with_weak_synaptic_currents":
                    #                     if down.down_judge(param_set, sanbool)==1:
                    #                         pattern.name == "SWS_FEW_SPIKES"
                    #                         g_li[1][n] = 2
                    #                     else:
                    g_li[1][n] = 5
                    pre_flag = -1
                    # ex_flag=1
                elif pattern.name == "RESTING":
                    g_li[1][n] = 6
                    pre_flag = -1
                    # if param > -1.0 or param < 1.0:
                    #     ex_flag = 1
                elif pattern.name == "EXCLUDED":
                    g_li[1][n] = 7
                    pre_flag = -1
                    # if param > -0.5 or param < 0.5:
                    # ex_flag = 1
                elif pattern.name == "ERROR":
                    g_li[1][n] = 8
                    pre_flag = -1
                    # if param > -1.0 or param < 1.0:
                    # ex_flag = 1
#                 print("pattern: ", pattern.name)

            except:
                print("Error, g", g)
                traceback.print_exc()

        np.save(save_p, g_li)
        print("save: {}".format(save_p))
    except:
        print("Error, original_g")
        traceback.print_exc()
    return "core"+str(core_n)+"_"+"param"+str(param_n), bifur_c, bifur_ch, s_flag, v_flag, ex_flag


def multiprocess(ncore,  T, dt, offset, savedir, date, path, cores, model_name, g_str, ave_con, step, param_range) -> None:
    p = os.getcwd()

    res_p = p + "/results_bifurcation/"+date+"_" + \
        model_name+"/"+date + "_" + model_name + "_"+g_str
    try:
        os.makedirs(res_p)
    except FileExistsError:
        print("file exist")
        pass

    jobs = []
    manager = multiprocessing.Manager()
    df_all = manager.list()
    results = []
    args = []
    for core in range(cores):
        args.append((df_all, T, dt, offset, res_p, savedir, date,
                    path, core, model_name, g_str, ave_con, step, param_range))

#     print("bifurcation analysis core".format(core))
    with Pool(processes=ncore) as pool:
        res = pool.map(bifur_total_multi, args)
        results.append(res)

    pd_bifur_T = pd.concat(df_all)
    # syuukei file save
    pd_bifur_T.to_excel(res_p + "/"+"bifurcation_Total.xlsx")

    print("bifurcation to wake {}".format(len(pd_bifur_T)))

    return results


def bifur_total_multi(args):
    df_all, T, dt, offset, res_p, savedir, date, path, core, model_name, g_str, ave_con, step, param_range = args

    save_ps = []
    bifur_cs = []
    bifur_chs = []
    s_flags = []
    v_flags = []
    ex_flags = []
    c = 0

#     if core ==0 or core ==1:
    fpath = savedir + date+"_"+model_name + \
        "/"+path + "_"+str(core) + ".pickle"
    print("file_path", fpath)
    with open(fpath, "rb") as f:
        iters = pickle.load(f)
        params = pickle.load(f)

    for i in range(len(params)):
        #             print("core{}, param{}".format(n,i))
        c += 1
#             if i!=4:
#                 continue
        pd_param = params.iloc[i, 0:]
        param_set = pd_param.to_dict()
        # print("param_set", param_set)

        save_p, bifur_c, bifur_ch, s_flag, v_flag, ex_flag = bifur_ana(
            T, dt, offset, param_set, res_p, g_str, ave_con, step, param_range, core, i)

        if bifur_c > 0 or bifur_ch > 0:
            bifur_cs.append(bifur_c)  # WAKE +
            bifur_chs.append(bifur_ch)  # WAKE +
            save_ps.append(save_p)
            s_flags.append(s_flag)
            v_flags.append(v_flag)
            ex_flags.append(ex_flag)

        if bifur_c > 0 and s_flag == 2 and v_flag == 1:
            print("core{}-param{}, Bifurcation!, AWAKE".format(core, i))

        if bifur_ch > 0 and s_flag == 5 or s_flag == 6:
            print("core{}-param{}, Bifurcation!, AWAKE_HIGH_FR".format(core, i))

        if (len(bifur_cs) + len(bifur_chs)) % 2 == 0 and (len(bifur_cs) + len(bifur_chs)) != 0:
            #     print("bifurcation_analysis core{} end".format(core))
            pd_bifur = pd.DataFrame(list(zip(save_ps, bifur_cs, bifur_chs, s_flags, v_flags, ex_flags)), columns=[
                                    "file", "wake_count", "wake_high_fr_count", "sleep_to_wake", "vmin", "exclude"])
            # syuukei file save
            pd_bifur.to_excel(
                res_p + "/"+"bifurcation_Total_core{}.xlsx".format(core))

    pd_bifur = pd.DataFrame(list(zip(save_ps, bifur_cs, bifur_chs, s_flags, v_flags, ex_flags)), columns=[
                            "file", "wake_count", "wake_high_fr_count", "sleep_to_wake", "vmin", "exclude"])
    # syuukei file save
    pd_bifur.to_excel(res_p + "/"+"bifurcation_Total_core{}.xlsx".format(core))

    if c > 0:
        per_bifur = len(bifur_cs)/c * 100
        per_bifurh = len(bifur_chs)/c * 100
    else:
        per_bifur = 0
        per_bifurh = 0
    print("core:{}, total_params:{}".format(core, c))
    print("core:{}, num of bifurcated by wake {}".format(core, len(bifur_cs)))
    print("core:{}, bifurcation to wake% {}".format(core, per_bifur))
    print("core:{}, num of bifurcated by wake with high FR {}".format(
        core, len(bifur_chs)))
    print("core:{}, bifurcation to wake with high FR% {}".format(core, per_bifurh))

    df_all.append(pd_bifur)
    return pd_bifur

#plot bifurcated waves
def vtrace_bifur(T, dt, offset, root_dir, fname, path, ex_path, ave_cons, exs, sanA_bool, g_str):
    time = np.arange(0, T+offset, dt)/1000  # s
    df_bifur = pd.read_excel(ex_path)
    # df for params bifurcated to WAKE
    df_bifur = df_bifur[df_bifur["wake_count"] > 0]
    df_bifur = df_bifur[(df_bifur["wake_count"] > 0) & (df_bifur["sleep_to_wake"] == 2) & (
        df_bifur["vmin"] == 1) & (df_bifur["exclude"] == 0)]  # df for params bifurcated to WAKE

    cps = []
    for i in df_bifur["file"]:
        i = i.replace("core", "")
        i = i.replace("param", "")
        cps.append((int(i.split("_")[0]), int(i.split("_")[1])))

    print("num of param_set bifurcated", len(cps))
    # print(cps)
    for n, cp in enumerate(cps):
        #     if n>1:
        #         break
        print("order", n)
        print("core:{}, param:{}".format(cp[0], cp[1]))
        fpath = root_dir + fname + "/"+path + "_"+str(cp[0]) + ".pickle"
        # print("file_path", fpath)
        with open(fpath, "rb") as f:
            iters = pickle.load(f)
            params = pickle.load(f)
            pd_param = params.iloc[cp[1], 0:]

            param_set = pd_param.to_dict()

    #         param_set = {'g_leak': 0.10623565193170517, 'g_nav': 3.9716793045043635, 'g_kvhh': 1.2637995931312185, 'g_kva': 0.03797592345948583, 'g_cav': 0.6461455466282616, 'g_kca': 7.137221898602252, 'g_nap': 0.14443157227422168, 'g_ampar': 0.1940482242203212, 't_ca': 22.324222527232898}

            print(param_set)
            ori = param_set[g_str]

            for ave_con in ave_cons:
                param_set_con = param_set.copy()
                ori_con = ori * ave_con
    #             print("average connection", ave_con)

                san_x = anmodel.models.Xmodel(channel_bool=sanA_bool)

                for ex in exs:
                    print("average connection", ave_con)
                    print("ex", ex)
                    param_set_con[g_str] = ori_con*10**(ex)
                    san_x.set_params(param_set_con)
                    print(g_str, param_set_con[g_str])

                    try:
                        #                     s = san_x.Solvers_sanA(dt, T+offset)
                        s, info = san_x.run_odeint(
                            int(1/dt*1000), int((T+offset)/1000))
                        v = s[int(T/dt/2)-1:, 0]

                        wavech = anmodel.analysis.WaveCheck(1000)
                        pattern = wavech.pattern(v, T, dt, pr=True)
                        print(pattern)
                        plt.figure(figsize=(15, 5))

                        plt.plot((time)[int(T/dt/2)-1:], v)
                        plt.xlabel('Time (sec)', fontsize=fs)

                #             plt.xlim(0,2)
                        plt.ylim(-110, 40)
                        plt.xticks(fontsize=fs)
                        plt.yticks(fontsize=fs)

                        plt.show()
                    except:
                        print("Error")
                        traceback.print_exc()

        f = bifur_fol + "/" + "core{}_param{}".format(cp[0], cp[1])


def saving_params_graphs(T, dt, offset, orders, exs, root_dir, date, model_name, g_str, all=True):
    bifur_fol = root_dir + "/results_bifurcation/"+date + \
        "_"+model_name + "/"+date+"_"+model_name+"_"+g_str
    bifur_total_data = bifur_fol+"/"+"bifurcation_Total.xlsx"
    print(bifur_total_data)
    time = np.arange(0, T, dt)

    fname = "/results/SWS_params"  # random_search
    path = date + "_" + model_name+"/" + "SWS_"+date  # random_search

    df_bifur = pd.read_excel(bifur_total_data)
    df_bifur = df_bifur[(df_bifur["wake_count"] > 0) & (df_bifur["sleep_to_wake"] == 2) & (
        df_bifur["vmin"] == 1) & (df_bifur["exclude"] == 0)]  # df for params bifurcated to WAKE

    cps = []
    for i in df_bifur["file"]:
        i = i.replace("core", "")
        i = i.replace("param", "")
        cps.append((int(i.split("_")[0]), int(i.split("_")[1])))
    print("all params", len(cps))

    if len(orders) == 0:
        orders = range(0, len(cps))
    print("all orders", len(orders))

    pdf = PdfPages(bifur_fol+"/multifig.pdf")
    pd_all = []

    for n, order in enumerate(orders):
        cp = cps[order]
        fpath = root_dir + fname + "/"+path + "_"+str(cp[0]) + ".pickle"
        print("file_path", fpath)
        with open(fpath, "rb") as f:
            iters = pickle.load(f)
            params = pickle.load(f)
            pd_param = params.iloc[cp[1], 0:]
            pd_all.append(pd_param)  # save params

            # gsave graphs
            param_set = pd_param.to_dict()
            print(param_set)
            ori = param_set[g_str]

            for ave_con in ave_cons:
                param_set_con = param_set.copy()
                ori_con = ori * ave_con
    #             print("average connection", ave_con)

                san_x = anmodel.models.Xmodel(channel_bool=sanA_bool)

                # axを展開して2次元から1次元に(2*2=4の配列になる)

                for m, ex in enumerate(exs):
                    fig, axes = plt.subplots()

                    print("average connection", ave_con)
                    print("ex", ex)
                    param_set_con[g_str] = ori_con*10**(ex)
                    san_x.set_params(param_set_con)
                    print(g_str, param_set_con[g_str])

                    try:
                        s, info = san_x.run_odeint(
                            int(1/dt*1000), int((T+offset)/1000))
                        v = s[int(T/dt/2)-1:, 0]
                        wavech = anmodel.analysis.WaveCheck(1000)
                        pattern = wavech.pattern(v, T, dt, pr=True)
                        print(pattern)
                        axes.plot((time)[int(T/dt/2)-1:], v, rasterized=True)
                        if m == 0:
                            # テキストの追加
                            fig.suptitle(param_set, fontsize=2)
                            axes.set_title("model: {}_{}_{}, order: {}, exp: {}".format(
                                date, model_name, g_str, order, ex))
                        else:
                            axes.set_title("model: {}_{}_{}, order: {}, exp: {}".format(
                                date, model_name, g_str, order, ex))
                        axes.set_xlabel('Time (sec)')
                        axes.set_ylim(-110, 40)
                        pdf.savefig(fig)

                    except:
                        print("Error")
                        traceback.print_exc()
    pdf.close()
    if len(pd_all) != 0:
        param_df = pd.concat(pd_all, axis=1).T
    # print(param_df)
        if all == True:
            param_df.to_excel(bifur_fol+"/bifurcation_all.xlsx")
        else:
            param_df.to_excel(bifur_fol+"/bifurcation_checked.xlsx")
    else:
        print("num of bifurcations is None")



if __name__ == "__main__":
    # search parameters for SWO
    if param_search == 1:

        sanA_random = anmodel.search.RandomParamSearch("X", 'SWS', 'SWS_HIGH_FR', cores,
                                                       time_h, 1000, 20, sanA_bool, model_name)
        sanA_random.multi_singleprocess(T, dt, offset, avecon, date)


     # bifurcation_analysis
    if bifur_analysis == 1:
        result_li = multiprocess(ncore, T, dt, offset, savedir, date,
                                 path, cores, model_name, g_str, avecon, step, param_range)
    #save results of bifurcation analysis
    if collect_bifur_params == 1:
        saving_params_graphs(T_c, dt, offset_c, orders, exs,
                             root_dir,  date, model_name, g_str, True)
        
    #select numbers from pdf of waveforms that bifurcated from wake-like to sleep-like patterns
    if select_numbers == 1:
        acr_path = "evince"  #Linux
        # acr_path = "C:\Program Files\Adobe\Acrobat DC\Acrobat\Acrobat.exe"  #windows
        
        pdf_path = bifur_fol + "/multifig.pdf"

        # open PDF
        pdf_pro = subprocess.Popen([acr_path, "", pdf_path], shell=False)
        orders = list(
            map(int, input("please input numbers of model that bifurcates (e.g. 2 3 5) and press \"enter\" button\n").split()))
        print(orders)

        pdf_pro.kill()

        saving_params_graphs(T_c, dt, offset_c, orders, exs,
                             root_dir, date, model_name, g_str, False)

  



from datetime import datetime
import os
import sys
import glob
import shutil
import traceback
import numpy as np
import pandas as pd

import pickle

# automatically obtain parameter sets and results that bifurcate from wake-like to seep-like patterns after 1st bifurcation analysis
# python3 auto_1st_bifurcation_check.py nmdar


def collect_bifurs(model_pre, bifur_g, p_th, date_base_str,date_up_str, model_op, bifur_auto=True):
    NE = 64  # number of excitatory neurons
    NI = 16   # number of inhibitory neurons
    con_M = 1.0  # mean for the lognormal distribution of the number of synapses per excitatory neuron
    con_M_in =  1.0 # mean for the lognormal distribution of the number of synapses per inhibitory neuron
    con_ex_ex = 0.01  # SD for the lognormal distribution of the number of synapses in Ex-Ex, Ex-in connections
    con_in_in = 0.01 # SD for the lognormal distribution of the number of synapses in In-In, In-Ex connections
    T =2500  #ms   simulation time

    postfix = "NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}".format(NE,NI,con_M,con_ex_ex,con_M_in,con_in_in,T)
    
    cv_th_w2=1.0
    cv_th_s2=1.3

    g_str = "g"+bifur_g
    root_dir = "results_bifurcation/"
    search_dir = "results/SWS_params/"
    files = os.listdir(root_dir)
    date_base_t = datetime.strptime(date_base_str, "%Y_%m_%d")
    date_up_t = datetime.strptime(date_up_str, "%Y_%m_%d")
    cc_dir = "cc_dir/"
    cc_dir_bifur = "cc_dir_bifur/"
    os.makedirs(cc_dir_bifur, exist_ok = True)
    
    savedir_col = "net_col/"
    os.makedirs(savedir_col, exist_ok = True)

    date_li =[]
    total_search_li=[]
    total_bifur_net_li=[]
    total_sws_li=[]
    total_single_bifur_li=[]
    
    total_sws_rate_li = []  #SWS/total
    total_bifur_single_rate_li = []
    sws_bifur_rate_li = []
    total_bifur_net_rate_li = []
    single_bifur_net_rate_li = []


    df_all=[]
    col_f=[]
    mid = []

    model_li=[]
    search_num_li = []
    sws_num_li = []
    sws_rate_li = []
    bifur_num_li = []


    for n, f in enumerate(files):
    #     if n>0:
    #         continue
        try:
            f_li = f.split("_")
            date_str = f_li[0] +"_"+ f_li[1]+"_" + f_li[2]
            date_t = datetime.strptime(date_str, "%Y_%m_%d")
            model_name_i = f_li[4]
            if date_t > date_base_t and date_t < date_up_t and model_pre == model_name_i:
                col_f.append(f)
                mid.append(date_str+"_" + f_li[3])
        except:
            print("error")
    
    print(col_f)


    for n, f in enumerate(col_f): 
            print("model_file: ", f)
            bifur_fol = root_dir + f +"/"+ f +"_"+g_str + "/"
            bifur_check_f  = bifur_fol + "bifurcation_checked.xlsx"  #single neuron bifurcation file
            if os.path.exists(bifur_check_f):
                params_bifur = pd.read_excel(bifur_check_f, engine="openpyxl", index_col=0)
                bifur_num_li.append(len(params_bifur))
            else:
                print("single_neuron_bifurcation_not_found:{}".format(f +"_"+g_str))
                params_bifur=[]
            for cp in range(len(params_bifur)):
#                 
                pars_df = params_bifur.iloc[cp,:]
                pars_df2 = pd.DataFrame(list(pars_df.values)).T
                pars_df2.columns = [list(pars_df.index)]
#                 print(pars_df)
                order = mid[n]+"_" + str(cp)
#                 print("order: ", order)
        #     print("net_files", net_files)
        #     print("model_order", orders)
                date_li.append(order)

        #         network structure dataframe
                net_st = {}
                net_st["NE"] = NE  # number of excitatory neurons
                net_st["NI"] = NI   # number of inhibitory neurons
                net_st["con_M"] = con_M  # mean for the lognormal distribution of the number of synapses per excitatory neuron
                net_st["con_M_in"] =  con_M_in # mean for the lognormal distribution of the number of synapses per inhibitory neuron
                net_st["con_ex_ex"] = con_ex_ex  # SD for the lognormal distribution of the number of synapses in Ex-Ex, Ex-in connections
                net_st["con_ex_in"] =con_ex_ex  # SD for the lognormal distribution of the number of synapses in Ex-Ex, Ex-in connections
                net_st["con_in_ex"] = con_in_in
                net_st["con_in_in"] =con_in_in # SD for the lognormal distribution of the number of synapses in In-In, In-Ex connections
                net_st["T"] =T  #ms   simulation time
                bifur_net = pd.DataFrame(list(net_st.values())).T
                bifur_net.columns= [list(net_st.keys())]
                bifur_net["bifur"]=0


                for i in range(len(bifur_net)):
        
                    model_name = model_pre +model_op +bifur_g

                    cc_name = "{}/{}_NE{}_NI{}_conM{}_SD{}_conMin{}_inSD{}_T{}".format(model_name, order, NE,NI,con_M,con_ex_ex,con_M_in,con_in_in,T)

                    try:
#                         
                        if not os.path.exists(cc_dir+cc_name+"_cc.xlsx"):
                            print("{} not found".format(cc_name))
                            continue
#         
                        cc_df = pd.read_excel(cc_dir+cc_name+"_cc.xlsx")
#                        
                        if bifur_auto == True:
#   
                            if os.path.exists(cc_dir+cc_name+"_cc.xlsx"):
                                p_ths_in = p_ths_in=(cc_df[(cc_df["%sleep"]>=30)&(cc_df["cv"]>=cv_th_s2)].index).tolist()
                            print("index sleep: ", p_ths_in)

                            if len(p_ths_in)!=0:
                                wake_in = 0 #np.argmin(np.where(cv_li==0,1000,cv_li))
                                sleep_in = 0 
                                bifur_flag = 0
                                for p_i in p_ths_in:
#                                     if cc_df["%wake"][p_i] > 30:
#                                         continue
                                    p_i_flag =0
                                    if p_i-1>=0:
                                        for j in range(p_i):
                                            if p_i-j-1>=0:
                                                if cc_df["cv"][p_i-j-1]==0:  #abnormal waveform between sleep and wake
                                                    p_i_flag==1
                                                    break #for j
                                                else:
#                                                     if cc_df["p"][p_i-j-1]> p_th:  #may be desynchro
#                                                     if os.path.exists(cc_dir+cc_name+"_cc2.xlsx"):
                                                        if cc_df["cv"][p_i-j-1]< cv_th_w2:
                                                            if cc_df["%wake"][p_i-j-1] > 30:
                                                                print("sleep cv: ", cc_df["cv"][p_i])
                                                                print("%sleep: ", cc_df["%sleep"][p_i])
                                                                print("wake cv: ", cc_df["cv"][p_i-j-1])
                                                                print("%wake: ", cc_df["%wake"][p_i-j-1])
                                                                wake_in = p_i-j-1
                                                                sleep_in = p_i
                                                                bifur_flag = 1
                                                                break  #for j
#                                                   
                                                        
                                            
                                    if bifur_flag ==1:
                                        bifur_net["bifur"]=1
                                        print("{} {} bifurcation!".format(model_name, order))
                                        if not os.path.exists(cc_dir_bifur+model_name+"/"):
                                            os.makedirs(cc_dir_bifur+model_name+"/")
                                    
#                                         shutil.copy(cc_dir+cc_name+"_cc2.xlsx",cc_dir_bifur+cc_name+"_cc2.xlsx")
                                        break  #p_i
#                                


                            else:
                                wake_in = 0 
                                sleep_in = 0 

                            
                            bifur_net["exp_w"]=np.array(cc_df["exp"])[wake_in]
    #                         print("exp_w", bifur_net["exp_w"][i])
                            bifur_net["exp_s"] = np.array(cc_df["exp"])[sleep_in]
                            desyn_sp_m = np.array(cc_df["mean_spike_fre"])[wake_in]
                            desyn_sp_var = np.array(cc_df["var_spike_fre"])[wake_in]
                            desyn_sp_m_ex = np.array(cc_df["mean_spike_fre_ex"])[wake_in]
                            desyn_sp_var_ex = np.array(cc_df["var_spike_fre_ex"])[wake_in]
                            syn_sp_m = np.array(cc_df["mean_spike_fre"])[sleep_in]
                            syn_sp_var = np.array(cc_df["var_spike_fre"])[sleep_in]
                            syn_sp_m_ex = np.array(cc_df["mean_spike_fre_ex"])[sleep_in]
                            syn_sp_var_ex = np.array(cc_df["var_spike_fre_ex"])[sleep_in]

                        df_unit = pd.concat([pars_df2, bifur_net],axis=1)
                        
                        print(df_unit)
#                         print("df_unit[NE]", df_unit["NE"])
                        df_unit["model"] = model_pre
                        df_unit["order"] = order
                        df_unit["mean_spike_fre_s"]= syn_sp_m
                        df_unit["var_spike_fre_s"]= syn_sp_var
                        df_unit["mean_spike_fre_w"]= desyn_sp_m
                        df_unit["var_spike_fre_w"]= desyn_sp_var
                        df_unit["cv_s"]= cc_df["cv"][sleep_in]
                        df_unit["cv_w"]= cc_df["cv"][wake_in]
                        df_unit["p"]= cc_df["p"][wake_in]

                        df_unit["mean_spike_fre_s_ex"]= syn_sp_m_ex
                        df_unit["var_spike_fre_s_ex"]= syn_sp_var_ex
                        df_unit["mean_spike_fre_w_ex"]= desyn_sp_m_ex
                        df_unit["var_spike_fre_w_ex"]= desyn_sp_var_ex
                        df_unit["cv_s_ex"]= cc_df["cv_ex"][sleep_in]
                        df_unit["cv_w_ex"]= cc_df["cv_ex"][wake_in]
                        df_unit["p_ex"]= cc_df["p_ex"][wake_in]


        #                 print(df_unit)
                        df_all.append(df_unit)
            
                        
                    except:
                        traceback.print_exc()

               # make unified DF


        # cal hit_rate
            if os.path.exists(bifur_fol):
                if len(os.listdir(bifur_fol))!=0:
                    pi_path = "SWS_"  + mid[n]  #_32.pickle"
                #     print("pickle_path", pi_path)
                    cores = len(glob.glob(search_dir+ f +"/" + "*.pickle"))
                    search_num, hit_num, hit_rate = cal_hit_rate(search_dir, f, pi_path,  cores)
                    search_num_li.append(search_num)
                    sws_num_li.append(hit_num)
                    sws_rate_li.append(hit_rate)

    df_unit_all = pd.concat(df_all, axis=0)
#     print(df_unit_all)
    df_unit_all.to_pickle(savedir_col + model_name+ "_"+postfix+ "_params.pkl")

    
    df_bifur_all=(df_unit_all[df_unit_all["bifur"]==1]).astype("float32")

#     df_bifur_all.reset_index(drop=True, inplace = True)
#     print(df_bifur_all)
    df_bifur_all.to_excel(savedir_col + model_name+ "_"+postfix+"_bifur_params.xlsx")

    bifur_net_sum = len(df_unit_all[df_unit_all["bifur"]==1])
#     bifur_net_sum = 132
    total_search_num=np.sum(search_num_li)
    total_sws_num = np.sum(sws_num_li)
    total_single_bifur_num = np.sum(bifur_num_li)
    if total_search_num !=0:
        total_sws_rate = total_sws_num/total_search_num*100  #SWS/total
        total_bifur_single_rate =total_single_bifur_num/total_search_num*100
        total_bifur_net_rate = bifur_net_sum/total_search_num*100
    else:
        total_sws_rate=0
        total_bifur_single_rate=0
        total_bifur_net_rate = 0
    if total_sws_num !=0:
        sws_bifur_rate = total_single_bifur_num/total_sws_num*100
    else:
        sws_bifur_rate=0
    if  total_single_bifur_num !=0:
        single_bifur_net_rate = bifur_net_sum/total_single_bifur_num*100
    else:
        bifur_num_li=0
    print("Search Total: ",total_search_num)
    print("num of SWS: ", total_sws_num)
    print("num of bifurcation (single): ", total_single_bifur_num)
    print("num of bifurcation (network): ", bifur_net_sum)
    print("SWS/Total", total_sws_rate)
    print("bifurcation (single neuron)/Total %", total_bifur_single_rate)
    print("bifurcation (single neuron)/SWS %", sws_bifur_rate)
    print("bifurcation (network)/Total %", total_bifur_net_rate)
    print("bifurcation (network)/bifurcation (single neuron) %", single_bifur_net_rate)


    model_li.append(model_pre)
    total_search_li.append(total_search_num)
    total_sws_li.append(total_sws_num)
    total_single_bifur_li.append(total_single_bifur_num)
    total_bifur_net_li.append(bifur_net_sum)

    total_sws_rate_li.append(total_sws_rate)  #SWS/total
    total_bifur_single_rate_li.append(total_bifur_single_rate)
    sws_bifur_rate_li.append(sws_bifur_rate)
    total_bifur_net_rate_li.append(total_bifur_net_rate)
    single_bifur_net_rate_li.append(single_bifur_net_rate)

    df_col = pd.DataFrame(list(zip(model_li, total_search_li, total_sws_li, total_single_bifur_li, total_bifur_net_li, total_sws_rate_li, total_bifur_single_rate_li, sws_bifur_rate_li, total_bifur_net_rate_li, single_bifur_net_rate_li)), columns=["Model", "Search_Total", "SWS","Bifurcation (single)", "Bifurcation (network)", "SWS/Total (%)","Bifurcation (single neuron)/Total (%)","Bifurcation (single neuron)/SWS (%)","Bifurcation (network)/Total (%)","Bifurcation (network)/Bifurcation (single neuron) (%)"])
    # print(df_col)
    df_col.to_excel(savedir_col + model_name+ "_"+postfix+ "_total.xlsx")
    return df_col

def cal_hit_rate(savedir, fname, path, cores):
    
    total_p = 0
    params_c = 0
    f_root = savedir + fname
            #print("file_path", fpath)
    if os.path.exists(f_root):

        for n in range(cores):
        #         print("core", n)
        #         if n > 1:
        #             break
                fpath = f_root +"/"+path + "_"+str(n) + ".pickle"
                #print("file_path", fpath)
    #             if os.path.exists(fpath):

                with open(fpath, "rb") as f:
                    iters = pickle.load(f)
                    params = pickle.load(f)
                total_p += iters
                params_c += len(params)

        if total_p ==0:
            hit_rate = 0
        else:
            hit_rate = params_c/total_p*100

#         print("Total", total_p)
#         print("the number of hits", params_c)
#         print("Hit rate", hit_rate)
        return total_p, params_c, hit_rate
    else:
        print("{} not found. Can,t calculate hit_rate".format(f_root))
        return 0, 0, 0

args = sys.argv


bifur_g = "_"+(args[1])
model_pre = "AN"   
model_op="_net_bifurcation"  
date_base_str = "2023_1_1"
date_up_str="2024_1_1"
p_th=0.01


bifur_auto = True



if __name__ == "__main__":
   df_col = collect_bifurs(model_pre, bifur_g, p_th, date_base_str, date_up_str, model_op, bifur_auto=True)

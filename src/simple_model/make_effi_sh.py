import os
import sys
import numpy as np
import pandas as pd

if __name__=="__main__":
    args = sys.argv
    
    lr_p = args[2]
    lr = args[1]
    offset = int(args[3]) # calc from offset order_num
    end = args[4]
    mth = args[5]
    con = args[6]

    # blockdim= 32
    p = os.getcwd()
    ndir = p + "/" 

    try:
        os.makedirs(ndir)
    except FileExistsError:
        print("{} is already exist".format(ndir))
        
        
        
    argss = "$1 $2 $3 $4 $5 $6 $7 $8 $9"



    pd_params = pd.read_excel(lr+"_"+lr_p+".xlsx",engine="openpyxl")
    sh_f = lr + "_"+lr_p

    if len(pd_params) != 0:
    #     print(lr_li[lr_in])

        savetxt_w = ndir + "/" + sh_f+".sh"
        with open(savetxt_w, "w") as f:
            offset_w =0
            for i in range(len(pd_params)):
                pd_i = pd_params.iloc[i,:]
                par_i = pd_i.to_dict()
                th_p = par_i["th_p"]
                th_d = par_i["th_d"]
                gp = par_i["gp"]
                gd = par_i["gd"]
                tau_ca_pre = par_i["tau_ca_pre"]
                tau_ca_post = par_i["tau_ca_post"]
                sig = par_i["sig"]
                tau_s = par_i["tau_s"]
                # lr = lr_li[int(par_i["lr"])]

                th_p_s = str(th_p)
                th_d_s = str(th_d)
                gp_s = str(gp)
                gd_s = str(gd)
                tau_pre_s = str(tau_ca_pre)
                tau_post_s = str(tau_ca_post)
                sig_s = str(sig)
                tau_s_s = str(tau_s)
                order = str(i+offset_w)


                file1 = th_p_s + " " + th_d_s + " " + gp_s + " " + gd_s + " " + tau_pre_s + " " + tau_post_s + " " + sig_s + " " + tau_s_s + " " + lr + " " + order +" " + argss
    #             print(file1)

                f.write(file1+'\n')

    param_f = lr+"_"+lr_p + ".sh"

    if end == "None":

        if mth == "GPU":
            if con != "r": 
                out_f1 = lr +"_"+lr_p+ "_GPU_w_off" +  str(offset) +".sh"
                out_f2 = lr + "_"+lr_p+ "_GPU_s_off"+  str(offset)+".sh"
                out_com1 = "./effi_G_w" 
                out_com2 = "./effi_G_s" 
            else:
                out_f1 = lr +"_"+lr_p+ "_GPU_r_w_off" +  str(offset) +".sh"
                out_f2 = lr + "_"+lr_p+ "_GPU_r_s_off"+  str(offset)+".sh"
                out_com1 = "./effi_G_r_w" 
                out_com2 = "./effi_G_r_s" 
        else:
            if con != "r": 
                out_f1 = lr +"_"+lr_p+ "_CPU_w_off" +  str(offset) +".sh"
                out_f2 = lr + "_"+lr_p+ "_CPU_s_off"+  str(offset)+".sh"
                out_com1 = "python3 effi_C_w.py" 
                out_com2 = "python3 effi_C_s.py" 
            else:
                out_f1 = lr +"_"+lr_p+ "_CPU_r_w_off" +  str(offset) +".sh"
                out_f2 = lr + "_"+lr_p+ "_CPU_r_s_off"+  str(offset)+".sh"
                out_com1 = "python3 effi_C_r_w.py" 
                out_com2 = "python3 effi_C_r_s.py" 

        with open(ndir+param_f) as reader, open(ndir+out_f1, 'w') as writer:
            writer.write("#!/bin/bash\n")
            for n, line in enumerate(reader):
                if n< offset:
                    continue
                # print(line)
                line_new = out_com1+" "+line
                # print(line_new)
                writer.write(line_new)


        with open(ndir+param_f) as reader, open(ndir+out_f2, 'w') as writer:
            writer.write("#!/bin/bash\n")
            for n, line in enumerate(reader):
                if n< offset:
                    continue

                line_new = out_com2+" "+line
                writer.write(line_new)

    else:

        if mth == "GPU":
            if con!="r":
                out_f1 = lr +"_"+lr_p+ "_GPU_w_off" +  str(offset) + "-" + end+".sh"
                out_f2 = lr + "_"+lr_p+ "_GPU_s_off"+  str(offset)  + "-" + end+".sh"
                out_com1 = "./effi_G_w" 
                out_com2 = "./effi_G_s" 
            else:
                out_f1 = lr +"_"+lr_p+ "_GPU_r_w_off" +  str(offset) + "-" + end+".sh"
                out_f2 = lr + "_"+lr_p+ "_GPU_r_s_off"+  str(offset)  + "-" + end+".sh"
                out_com1 = "./effi_G_r_w" 
                out_com2 = "./effi_G_r_s" 
                
        else:
            if con!="r":
                out_f1 = lr +"_"+lr_p+ "_CPU_w_off" +  str(offset)  + "-" + end +".sh"
                out_f2 = lr + "_"+lr_p+ "_CPU_s_off"+  str(offset) + "-" + end +".sh"
                out_com1 = "python3 effi_C_w.py" 
                out_com2 = "python3 effi_C_s.py" 
            else:
                out_f1 = lr +"_"+lr_p+ "_CPU_r_w_off" +  str(offset)  + "-" + end +".sh"
                out_f2 = lr + "_"+lr_p+ "_CPU_r_s_off"+  str(offset) + "-" + end +".sh"
                out_com1 = "python3 effi_C_r_w.py" 
                out_com2 = "python3 effi_C_r_s.py" 

        with open(ndir+param_f) as reader, open(ndir+out_f1, 'w') as writer:
            # 読み込み元の行ごとにイテレーション
            writer.write("#!/bin/bash\n")
            for n, line in enumerate(reader):
                if n< offset:
                    continue
                if n > int(end):
                    continue
                # 置換
                # print(line)
                line_new = out_com1+" "+line
                # print(line_new)
                # 書き出し
                writer.write(line_new)


        with open(ndir+param_f) as reader, open(ndir+out_f2, 'w') as writer:
            writer.write("#!/bin/bash\n")
            for n, line in enumerate(reader):
                if n< offset:
                    continue
                if n > int(end):
                    continue

                # 置換
        #         print(line)
                line_new = out_com2+" "+line
        #         print(line_new)
                # 書き出し
                writer.write(line_new)

         #  make total file 
    if end =="None":
        if mth == "GPU":
            if con!="r":
                total_f = lr +"_"+lr_p + "_effi_GPU_off" +  str(offset)+ ".sh"
            else:
                total_f = lr +"_"+lr_p + "_effi_GPU_r_off" +  str(offset)+ ".sh"
        else:
            if con!="r":
                total_f = lr +"_"+lr_p + "_effi_CPU_off" +  str(offset)+".sh"
            else:
                total_f = lr +"_"+lr_p + "_effi_CPU_r_off" +  str(offset)+".sh"

        print("sh " +total_f)
        
        
        # args ="$1 $2 $3 $4 $5 $6 $7 $8 $9"
        
            

        with open(ndir+total_f, 'w') as writer:
            writer.write("#!/bin/bash\n")
            writer.write("sh "+ out_f1 + " " + argss +" & \n")
            writer.write("sh "+ out_f2 + " " + argss +" & \n")
            writer.write("wait\n")

            writer.write("rm " + ndir + out_f1 + "\n")
            writer.write("rm " + ndir + out_f2 )

    else:
        if mth == "GPU":
            if con!="r":
                total_f = lr +"_"+lr_p + "_effi_GPU_off" +  str(offset)+"-"+end+  ".sh"
            else:
                total_f = lr +"_"+lr_p + "_effi_GPU_r_off" +  str(offset)+"-"+end+  ".sh"
        else:
            if con!="r":
                total_f = lr +"_"+lr_p + "_effi_CPU_off" +  str(offset)+"-"+end+  ".sh"
            else:
                total_f = lr +"_"+lr_p + "_effi_CPU_r_off" +  str(offset)+"-"+end+  ".sh"

        print("sh " +total_f)
        # NE = "10"
        # Tp = "60000"
        # dt = "0.05"
        # sw = "wake"
        # dsb = "ISI_40_{}".format(dt)

        # seed = "6"
        # args ="$1 $2 $3 $4 $5 $6 $7 $8 $9"

        with open(ndir+total_f, 'w') as writer:
            writer.write("#!/bin/bash\n")
            writer.write("sh "+ out_f1 + " " + argss +" & \n")
            writer.write("sh "+ out_f2 + " " + argss +" & \n")
            writer.write("wait\n")

            writer.write("rm " + ndir + out_f1 + "\n")
            writer.write("rm " + ndir + out_f2 )
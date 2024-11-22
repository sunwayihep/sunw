#!/usr/bin/python

import os
import sys
import socket
import numpy as np

import insert_point
# Nc = 6
# Ns = 16
# Nt = 16
max_traj = 2000

pattern = 'Plaquette: <'



# 检查文件夹存在，不存在则创建
# 检查文件夹是否存在
def check_mkdir (folder_path):
    if not os.path.exists(folder_path):  # 如果文件夹不存在，则创建它
        os.makedirs(folder_path)
        print(f"folder path '{folder_path}' added.")
    else:
        print(f"folder path '{folder_path}' already exists.")

# 检查文件存在，并且超过200行
def file_exists_and_over_200_lines(file_path):
    if os.path.isfile(file_path) and os.stat(file_path).st_size != 0:
        with open(file_path, 'r') as file:
            return len(file.readlines()) > 200
    return False


def run_a_round (const_lst: list, Nc, run_log_path, res_log_path, csv_dir, max_traj) :

    exec_proc_name = f'../heatbath_su{Nc}'

    print(f'#######  your config: color = {Nc}, Ns = {Ns}, Nt = {Nt}')

    const = np.array(const_lst)
    print(const)
    beta = const * Nc * Nc / 3
    print('#########\n', 'host_name = ', socket.gethostname(), '\n##############\n')
    for i, x in enumerate(beta):

        file_name = os.path.join(run_log_path, f'const_{const[i]}.txt')
        res_file_name = os.path.join(res_log_path, f'res_const_{const[i]}.txt')
        
        if file_exists_and_over_200_lines(file_name): 
            print(f'file {file_name} already exists')

        else :
            print(f'beta = {beta[i]} begin=====================')
            cmd = f'./{exec_proc_name} {Ns} {Nt} {beta[i]} {max_traj} > {file_name} 2>&1'
            print(cmd)
            os.system(cmd)

        # export res_log
        try:
            with open(file_name, "r") as file_name:
                lines = [line.strip() for line in file_name if line.startswith(pattern)]
                
            with open(res_file_name, "w") as out_file:
                out_file.write('\n'.join(lines) + '\n')
        except IOError as e:
            print(f"Error processing file {file_name}: {e}")

        # end    
        print(f'beta = {beta[i]} end =====================')


if __name__ == '__main__':
    if len(sys.argv) < 4: 
        print('you must input your color first')
        print("your cmd should be 'run.py color ns nt' ")
        sys.exit(-1)
    Nc = int(sys.argv[1])
    Ns = int(sys.argv[2])
    Nt = int(sys.argv[3])
    run_log_path = '../run_log/'
    res_log_path = '../res_log/'
    csv_dir = '../csv_dir/'

    threshold_x = 1e-4
    threshold_y = 2
    from my_csv import read_csv, generate_csv


    check_mkdir(run_log_path)
    check_mkdir(res_log_path)
    check_mkdir(csv_dir)
    const_lst = [
        0.6, 0.8, 1, 1.25, 1.5, 1.6, 1.7, 1.75, 1.8, 1.9, 1.92, 
        1.94, 1.96, 1.98, 1.985, 1.99, 1.995, 1.996, 1.997, 1.998, 
        1.999, 2, 2.02, 2.04, 2.044, 2.04401, 2.0442, 2.1, 2.25, 
        2.5, 3, 4, 10, 20, 40
    ]
    run_a_round(const_lst, Nc, run_log_path, res_log_path, csv_dir, max_traj=2000)
    
    csv_file = f'plaq_traj_{Nc}_{Ns}x{Nt}.csv'
    generate_csv (color=Nc, Ns=Ns, Nt=Nt, lst=const_lst, csv_path=csv_dir+csv_file, log_path=res_log_path)

    _, lambda_, plaquette_ = read_csv(csv_dir + csv_file)

    insrt = insert_point.interpolate(lambda_, plaquette_, threshold_x=threshold_x, threshold_y=threshold_y)
    print(f'insrt = {insrt}')
    while insrt:
        print('insert points: ', insrt)
        run_a_round(insrt, Nc, run_log_path, res_log_path, csv_dir, max_traj=5000)
        for elem in insrt:
            const_lst.append(elem)
        
        # repeat
        const_lst.sort()
        generate_csv (color=Nc, Ns=Ns, Nt=Nt, lst=const_lst, csv_path=csv_dir+csv_file, log_path=res_log_path)
        _, lambda_, plaquette_ = read_csv(csv_dir + csv_file)
        insrt = insert_point.interpolate(lambda_, plaquette_, threshold_x=threshold_x, threshold_y=threshold_y)

    with open('consts', 'w') as f:
        for elem in const_lst:
            f.write(str(elem) + '\n')

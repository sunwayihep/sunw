#!/usr/bin/python

import os
import sys
import socket
import numpy as np
import insert_point
from pathlib import Path
import sunw_name_generator


pattern = 'Plaquette: <'

# 检查文件夹存在，不存在则创建
from pathlib import Path

def check_mkdir(folder_path):
    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"folder path '{folder_path}' already exists or created.")

# 检查文件存在，并且超过指定行数
def file_exists_and_over_lines(file_path, min_lines=200):
    try:
        with open(file_path, 'r') as file:
            return sum(1 for _ in file) > min_lines
    except (OSError, IOError):
        return False

class Config:
    def __init__(self, proc_name: str, Nc: int, Ns: int, Nt: int, lambda_pair: list, traj_pair: list, file_path: dict):
        log_name_generator = sunw_name_generator.SunwLogFileName()
        csv_name_generator =  sunw_name_generator.SunwCsvFileName()

        self.proc_name = proc_name
        self.Nc = Nc
        self.Ns = Ns
        self.Nt = Nt
        self.lambda_pair = lambda_pair
        self.beta = [1 / lambda_pair[0] * Nc * Nc, 1 / lambda_pair[1] * Nc * Nc]
        self.traj_pair = traj_pair
        self.log_dir = file_path['log_dir']
        # self.res_path = file_path['res_path']
        self.csv_dir = file_path['csv_dir']
        self.log_file = log_name_generator.generate_log_filename(self.lambda_pair[0], self.traj_pair[0], self.lambda_pair[1], self.traj_pair[1])
        self.csv_file = csv_name_generator.generate_csv_filename(self.lambda_pair[0], self.traj_pair[0], self.lambda_pair[1], self.traj_pair[1])

        self.cmd = f'./{self.proc_name} {self.Ns} {self.Nt} {self.beta[0]} {self.beta[1]} {self.traj_pair[0]} {self.traj_pair[1]} > {self.log_dir}/{self.log_file} 2>&1'

    def run(self):
        import pandas as pd
        log_file_name = os.path.join(self.log_dir, self.log_file)
        csv_file_name = os.path.join(self.csv_dir, self.csv_file)
        
        print (f'##################### \n cmd {self.cmd} begin ......')
       
        if file_exists_and_over_lines(csv_file_name, self.traj_pair[0] + self.traj_pair[1]):
            print(f'file {csv_file_name} already has enough lines')
            df = pd.read_csv(csv_file_name)
            return df['plaquette'].values[-1]
        else:
            os.system(self.cmd)
            try:
                plaquettes = [
                    float(line.split()[-1])
                    for line in open(log_file_name, 'r') if line.startswith(pattern)
                ]
                
                # 构建 DataFrame
                lambda_values = [self.lambda_pair[0]] * self.traj_pair[0] + [self.lambda_pair[1]] * self.traj_pair[1]
                # beta_values = [self.beta[0]] * self.traj_pair[0] + [self.beta[1]] * self.traj_pair[1]
                traj_numbers = list(range(self.traj_pair[0])) + list(range(self.traj_pair[0], self.traj_pair[0] + self.traj_pair[1]))
                df = pd.DataFrame({
                    'lambda': lambda_values,
                    'traj': traj_numbers,
                    'plaquette': plaquettes
                })
                
                # 保存 CSV
                df.to_csv(csv_file_name, index=False)
                print(f'CSV has been saved to {csv_file_name}')
            except Exception as e:
                print(f"Error processing log file: {e}")
            print(f'##################### cmd {self.cmd} end ......')
            return plaquettes[-1]
            

# lambda_lst  =  [lambda1[], lambda2[]]
# traj_lst = [traj1, traj2]
# file_path = {log_path: xx, csv_path: xx}
def run_all (Nc: int, Ns: int, Nt: int, lambda_lst: list, traj_lst: list, file_path: dict) :
    
    print(f'#######  your config: color = {Nc}, Ns = {Ns}, Nt = {Nt}')
    
    exec_proc_name = f'../heatbath_su{Nc}'
    process_numbers = len(lambda_lst[0])

    plaquettes = []
    for i in range(process_numbers):
        lambda_pair = [lambda_lst[0][i], lambda_lst[1][i]]
        traj_pair = [traj_lst[0], traj_lst[1]]
        
        config = Config(exec_proc_name, Nc, Ns, Nt, lambda_pair, traj_pair, file_path)
        plaquettes.append(config.run())

    return plaquettes

def run_all_with_insert_point(Nc: int, Ns: int, Nt: int, lambda_lst: list, traj_lst: list, file_path: dict, interpolate_policy: dict) :
    plaquettes = run_all(Nc, Ns, Nt, lambda_lst, traj_lst, file_path)
    insert_points = insert_point.interpolate(lambda_lst[1], plaquettes, interpolate_policy['threshold_x'], interpolate_policy['threshold_slope'])

    while insert_points:
        insert_lambda = [[lbd if interpolate_policy['same_lambda'] else interpolate_policy['init_lambda'], lbd] for lbd in insert_points]
        plaquettes = run_all(Nc, Ns, Nt, insert_lambda, traj_lst, file_path)

        lambda_lst[0] += insert_lambda[0]
        lambda_lst[1] += insert_lambda[1]
        insert_lambda[0].sort()
        insert_lambda[1].sort()

        insert_points = insert_point.interpolate(lambda_lst[1], plaquettes, interpolate_policy['threshold_x'], interpolate_policy['threshold_slope'])

    return lambda_lst

if __name__ == '__main__':
    if len(sys.argv) < 4: 
        print('you must input your color first')
        print("your cmd should be 'run.py color ns nt' ")
        sys.exit(-1)
    print('#########\n', 'host_name = ', socket.gethostname(), '\n##############\n')

    Nc = int(sys.argv[1])
    Ns = int(sys.argv[2])
    Nt = int(sys.argv[3])
    run_log_dir = '../run_log/'
    csv_dir = '../csv_dir/'

    threshold_x = 1e-4
    threshold_slope = 2
    # from my_csv import read_csv, generate_csv

    init_points = 10
    max_lambda = 5
    lambda_ = [0 + max_lambda / init_points * i for i in range(0, init_points)]

    check_mkdir(run_log_dir)
    check_mkdir(csv_dir)
    lambda_lst = [[5, 4]] * 2
    traj_lst = [100, 100]
    file_path = {'log_dir': run_log_dir, 'csv_dir': csv_dir}

    interpolate_policy = {'threshold_x': 1e-4, 'threshold_slope': 2, 'same_lambda': True, 'init_lambda': 0.6}
    lambda_lst = run_all_with_insert_point(Nc, Ns, Nt, lambda_lst, traj_lst, file_path, interpolate_policy)
    print(lambda_lst)

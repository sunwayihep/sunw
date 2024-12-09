import subprocess
import os
import pandas as pd
from src.sunw_run_config import SunwConfig, LambdaTrajPair, ResPath

class RunSunw:
    def __init__(self, sunw_config_: SunwConfig):
        self.sunw_config_ = sunw_config_
    def run(self, step):
        # print(f'DEBUG: log_res_path = {os.path.dirname(self.sunw_config_.log_res_path_)}')
        # print(f'DEBUG: csv_res_path = {os.path.dirname(self.sunw_config_.csv_res_path_)}')
        # print(f'DEBUG: last_gauge_path = {os.path.dirname(self.sunw_config_.full_gauge_path_)}')
        if not os.path.exists(os.path.dirname(self.sunw_config_.log_res_path_)):
            os.makedirs(os.path.dirname(self.sunw_config_.log_res_path_))
        if not os.path.exists(os.path.dirname(self.sunw_config_.csv_res_path_)):
            os.makedirs(os.path.dirname(self.sunw_config_.csv_res_path_))
        if not os.path.exists(os.path.dirname(self.sunw_config_.full_gauge_path_)): 
            os.makedirs(os.path.dirname(self.sunw_config_.full_gauge_path_))

        log_file = os.path.join(self.sunw_config_.log_res_path_, 
                               f"log_step_{step}_lambda_{self.sunw_config_.lambda_}_traj_{self.sunw_config_.traj_}.log")
        csv_file = os.path.join(self.sunw_config_.csv_res_path_, 
                               f"res_step_{step}_lambda_{self.sunw_config_.lambda_}_traj_{self.sunw_config_.traj_}.csv")
        
        # 运行命令并捕获输出
        with open(log_file, 'w+') as log_out, open('error.log', 'w+') as err_out:
            process = subprocess.Popen(self.sunw_config_.cmd_, 
                                       stdout=log_out, 
                                       stderr=err_out,
                                       text=True)
            process.wait()
        
        # 提取plaquette值
        plaquettes = []
        with open(log_file, 'r') as log_in:
            for line in log_in:
                if line.startswith('Plaquette: <'):
                    try:
                        # 提取mean值，即最后一个数字
                        mean_value = line.split('mean:')[1].strip()
                        plaquette = float(mean_value)
                        plaquettes.append(plaquette)
                    except (IndexError, ValueError):
                        continue
        
        # 创建DataFrame并保存CSV
        if plaquettes:
            lambda_ = [self.sunw_config_.lambda_] * len(plaquettes)
            df = pd.DataFrame({
                'lambda': lambda_,
                'plaquette': plaquettes
            })
            df.to_csv(csv_file, index=False)

        # 返回最终的gauge文件二进制路径
        return self.sunw_config_.output_gauge_bin_, lambda_, plaquettes



def run_interface (
        process_path: str,
        dims: list,
        color: int,
        lambda_traj_pair: LambdaTrajPair,
        gauge_input_path: str,
        gauge_output_path: str,
        res_path: ResPath,
        rank: int
):

    run_config = SunwConfig(process_path, color, dims, lambda_traj_pair, 
                            gauge_input_path, gauge_output_path, res_path, rank)
    
    run_sunw = RunSunw(run_config)
    gauge_bin_path, lambda_, plaquettes = run_sunw.run(rank)
    return gauge_bin_path, lambda_, plaquettes


# def main():
def run_a_sequence(
        lambda_traj_pair_list: list,
        process_path: str,
        dims: list,
        color: int,
        gauge_input_path: str,
        gauge_output_prefix: str,
        log_path_prefix: str,
        csv_path_prefix: str
):

    from src.sunw_run_config import generate_lambda_traj_pair_prefix
    sub_dir = generate_lambda_traj_pair_prefix(lambda_traj_pair_list)
    sub_dir += f'dim{len(dims)}'
    for dim in dims:
        sub_dir += f'_{dim}'

    gauge_output_path = os.path.join(gauge_output_prefix, sub_dir)
    log_path = os.path.join(log_path_prefix, sub_dir)
    csv_path = os.path.join(csv_path_prefix, sub_dir)
    res_path = ResPath(log_path, csv_path)

    if not os.path.exists(gauge_output_path):
        os.makedirs(gauge_output_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    

    lambda_out_list = []
    plaquettes_out_list = []

    for i, pair in enumerate(lambda_traj_pair_list):
        gauge_bin_path, lambda_out, plaquettes_out = run_interface(process_path, dims, color, pair, gauge_input_path, gauge_output_path, res_path, i + 1)
        print(f'DEBUG: gauge_bin_path = {gauge_bin_path}')
        gauge_input_path = gauge_bin_path

        lambda_out_list += lambda_out
        plaquettes_out_list += plaquettes_out
    
    # 创建汇总的DataFrame并保存
    summary_df = pd.DataFrame({
        'lambda': lambda_out_list,
        'plaquette': plaquettes_out_list
    })
    summary_csv_path = os.path.join(csv_path, sub_dir + 'summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)


def main() :
    process_path = '/public2/home/ybyang/wjc/sunw_ndim/dim4/sunw_su6'
    Lx, Ly, Lz, Lt = 32, 32, 32, 32
    dims = [Lx, Ly, Lz, Lt]
    color = 6
    cur_abs_path = os.path.dirname(os.path.abspath(__file__))
    gauge_input_path = 'None' # 开始的时候没有gauge文件

    gauge_output_prefix = os.path.join(cur_abs_path, 'gauge')
    log_path_prefix = os.path.join(cur_abs_path, 'log')
    csv_path_prefix = os.path.join(cur_abs_path, 'csv')

    lambda_traj_pair_list = []
    lambda_traj_pair_list.append(LambdaTrajPair(lambda_=5, traj=200, save_per_n_traj=1000))
    lambda_traj_pair_list.append(LambdaTrajPair(lambda_=1.4835, traj=2000, save_per_n_traj=1000))
    lambda_traj_pair_list.append(LambdaTrajPair(lambda_=1.475, traj=2000, save_per_n_traj=1000))
    
    run_a_sequence(lambda_traj_pair_list, process_path, dims, color, gauge_input_path, gauge_output_prefix, log_path_prefix, csv_path_prefix)

if __name__ == '__main__':
    main()
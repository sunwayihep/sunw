import os

suffix = ".bin" # the suffix of the gauge file

class ResPath:
    def __init__(self, log_dir: str, csv_dir: str):
        self.log_dir_ = log_dir
        self.csv_dir_ = csv_dir
        # self.gauge_path_prefix_ = gauge_path_prefix

class LambdaTrajPair:
    def __init__(self, lambda_: float, traj: int, save_per_n_traj: int):
        self.lambda_ = lambda_
        self.traj_ = traj
        self.step_ = save_per_n_traj
    def __str__(self):
        return f"lambda_{self.lambda_}_traj_{self.traj_}"

'''
    @brief: 
        This function is used to get the process name
    @param:
        color: int, the color of the process
        dims: list, the dimensions of the process
    @return:
        process_name: str, the name of the process
'''
def get_process_name(color: int, dims: list):
    return f"heatbath_su{color}_nd{len(dims)}"


'''
    @brief: 
        This class is used to set the gauge file path
    @param:
        lambda_traj_pair_: LambdaTrajPair,
        gauge_path_prefix: str    format: 'prefix/step_x/', x means the step number
    @return:
        full_gauge_path_: str, format: 'prefix/step_x/lambda_traj_pair_step_xx'
'''
class GaugeConfig:
    def __init__(
        self,
        gauge_read_path: str,
        lambda_traj_pair_: LambdaTrajPair,
        gauge_path_prefix: str    # we need to add new path after the prefix
    ):
        self.full_gauge_prefix_ = os.path.join(gauge_path_prefix, str(lambda_traj_pair_))
        self.output_gauge_bin_ = self.full_gauge_prefix_ + suffix
        self.gauge_read_path_ = gauge_read_path

class SunwConfig:
    def __init__(
        self, 
        process_path_: str,  # use for naming the process
        color_: int,          # Nc 
        dims_: list,          # Nd = len(dims)
        lambda_traj_pair_: LambdaTrajPair,
        gauge_read_path : str,
        gauge_path_prefix: str,
        res_path_: ResPath, 
        rank : int            # the rank in a sequence of execution
    ):

        self.process_name_ = os.path.join(process_path_, get_process_name(color_, dims_))
        self.lambda_ = lambda_traj_pair_.lambda_
        self.beta_ = color_ * color_ / self.lambda_
        self.gauge_config_ = GaugeConfig(gauge_read_path, lambda_traj_pair_, 
                                        os.path.join(gauge_path_prefix, f"step_{rank}"))
        # generate the command
        # command format: ./heatbath_su{color}_nd{len(dims)} {dim1} {dim2} ... {dimN} {beta} {total_traj} [{save_per_n_traj} {input_dir} {output_dir}
        self.cmd_ = [self.process_name_]
        for dim in dims_:
            self.cmd_.append(str(dim))
        self.cmd_.append(str(self.beta_))
        self.cmd_.append(str(lambda_traj_pair_.traj_))
        self.cmd_.append(str(lambda_traj_pair_.step_)) # if you use my code, it is necessary to set parameter save_per_n_traj
        self.cmd_.append(self.gauge_config_.gauge_read_path_)            # if you use my code, it is necessary to input the gauge file
        self.cmd_.append(self.gauge_config_.full_gauge_prefix_)           # if you use my code, it is necessary to output the gauge file

        # log and res path
        self.log_res_path_ = res_path_.log_dir_
        self.csv_res_path_ = res_path_.csv_dir_
        # self.last_gauge_path_ = self.gauge_config_.last_gauge_path_
        self.full_gauge_path_ = self.gauge_config_.full_gauge_prefix_

        self.lambda_ = lambda_traj_pair_.lambda_
        self.traj_ = lambda_traj_pair_.traj_
        self.output_gauge_bin_ = self.gauge_config_.output_gauge_bin_
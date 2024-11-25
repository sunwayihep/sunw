import re

class SunwFileNameGenerator:
    def __init__(self):
        raise NotImplementedError("This class is abstract and cannot be instantiated directly.")

class SunwLogFileName(SunwFileNameGenerator):
    def __init__(self):
        pass
    def generate_log_filename(self, lambda1, traj1, lambda2, traj2):
        return f"log_lambda1_{lambda1}_traj1_{traj1}_lambda2_{lambda2}_traj2_{traj2}.log"
    def extract_log_params(self, filename):
        pattern = re.compile(
            r'log_lambda1_(?P<lambda1>\d+\.\d+)_traj1_(?P<traj1>\d+)_lambda2_(?P<lambda2>\d+\.\d+)_traj2_(?P<traj2>\d+)\.log'
        )
        match = pattern.match(filename)
        if match:
            return {
                'lambda1': float(match.group('lambda1')),
                'traj1': int(match.group('traj1')),
                'lambda2': float(match.group('lambda2')),
                'traj2': int(match.group('traj2')),
            }
        else:
            raise ValueError('Wrong filename format')


class SunwCsvFileName(SunwFileNameGenerator):
    def __init__(self):
        pass
    def generate_csv_filename(self, lambda1, traj1, lambda2, traj2):
        """ 
        根据给定的 lambda 和 traj 生成对应的文件名。
        """
        return f"csv_lambda1_{lambda1}_traj1_{traj1}_lambda2_{lambda2}_traj2_{traj2}.csv"

    def extract_csv_params(self, filename):
        """
        从文件名中提取 lambda 和 traj 参数。
        """
        pattern = re.compile(
            r'csv_lambda1_(?P<lambda1>\d+\.\d+)_traj1_(?P<traj1>\d+)_lambda2_(?P<lambda2>\d+\.\d+)_traj2_(?P<traj2>\d+)\.csv'
          )
        match = pattern.match(filename)
        if match:
            return {
                'lambda1': float(match.group('lambda1')),
                'traj1': int(match.group('traj1')),
                'lambda2': float(match.group('lambda2')),
                'traj2': int(match.group('traj2')),
            }
        else:
            raise ValueError('Wrong filename format')
    
if __name__ == "__main__":
    filename = "csv_lambda1_1.5_traj1_200_lambda2_2.0_traj2_400.csv"
    print(SunwCsvFileName().extract_csv_params(filename))

    filename = "log_lambda1_1.5500000000001_traj1_200_lambda2_2.0_traj2_400.log"
    print(SunwLogFileName().extract_log_params(filename))

    lambda1 = 2.3
    lambda2 = 4.5
    traj1 = 200
    traj2 = 400
    print(SunwCsvFileName().generate_csv_filename(lambda1, traj1, lambda2, traj2))
    print(SunwLogFileName().generate_log_filename(lambda1, traj1, lambda2, traj2))

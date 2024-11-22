import csv
import numpy as np
def read_csv(file_name):
  const_ = []
  lambda_ = []
  plaquette_ = []
  with open(file_name, "r", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      const_.append(float(row['const']))
      lambda_.append(float(row['lambda']))
      plaquette_.append(float(row['plaquette']))
  return const_, lambda_, plaquette_

def write_csv(file_name, data):
  with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

# read last column of each line in file
def get_last_column(file_name) -> np.array:
  data = []
  with open(file_name, 'r') as file:
    for line in file:
      # 使用split()函数以空格为分隔符分割每行，并取最后一个元素
      data.append(float(line.split()[-1]))
  return np.array(data)


def generate_data (color, const: np.array, log_path) :
    
  result = []
  result.append(['color', 'const', 'lambda', 'plaquette'])
  for i, const_elem in enumerate(const):
    file_path = f'{log_path}/res_const_{const[i]}.txt'
    
    data = get_last_column(file_path)
    if len(data) > 100:
      mean_val = sum(data[-100:]) / 100
      # print('DEBUG:########## mean_val = ', mean_val)
    elif len(data) > 20:
      mean_val = sum(data[-20:]) / 20
    else:
      mean_val = data[-1]
    # Nc, const, beta, lambda = 3/const
    result.append([color, const_elem, 3 / const_elem, mean_val])

  return result


def generate_csv (color: int, Ns : int, Nt : int, lst : list, csv_path, log_path) -> list:
  const = np.array(lst)
  result = generate_data(color, const, log_path=log_path)
  write_csv(csv_path, result)
  return lst



import matplotlib.pyplot as plt
import numpy as np

max_traj = 2000
Nc = 8
# read last column of each line in file
def get_last_column(file_name) -> np.array:
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            # 使用split()函数以空格为分隔符分割每行，并取最后一个元素
            data.append(float(line.split()[-1]))
            # print(line.split()[-1])
    return np.array(data)

def add_line (x, y, line_label) : 
    plt.plot(x, y, label=line_label, marker = 'o')


def draw_table (color, const: np.array, plt_max_x = max_traj, table_name = 'plaq_traj', data_path='',  save = False) :
    beta = (color * color / 3) * const
    plt.clf()
    plt.xlim(0, plt_max_x)    # x 坐标限制在 0 到 max_traj

    for i, beta_elem in enumerate(beta):
        res_file_name = data_path + f'res_const_{const[i]}.txt'
        res = get_last_column(res_file_name)[:plt_max_x]
        item_num = res.size
        plt.plot(np.arange(1, 1 + item_num), res, label=(r'$\beta = $' + str(beta_elem) + f' const = {const[i]}'), marker = 'None')
        # print(np.arange(1, 1 + item_num), res)

    plt.xlabel('traj')
    plt.ylabel('plaquette')
    #plt.legend(loc='upper right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05)) 
    plt.tight_layout()
    
    if save:
        plt.savefig(table_name)
    plt.show()


if __name__ == '__main__':
    data_path = '../res_log/'
    origin_const = []
    #     0.6, 0.8, 1, 1.25, 1.5, 1.6, 1.7, 1.75, 1.8, 1.9, 2, 2.02, 2.04, 2.044
    #     , 2.04401, 2.0442, 2.1, 2.25, 2.5, 3, 4, 10, 20, 40
    # ]

    with open('consts', 'r') as f:
        for line in f:
            origin_const.append(float(line))
    print(origin_const)
    const = [origin_const[i] for i, x in enumerate(origin_const)]# if x > 1.9 and x < 2.1 ]

    draw_table(Nc, np.array(const), 2000, 'plaquette_traj', data_path,True)

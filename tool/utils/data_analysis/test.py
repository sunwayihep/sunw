import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import os

def plot_plaquette(file_paths):
    # 创建一个 Streamlit 选择框来选择文件
    selected_files = st.multiselect('选择文件', file_paths)

    for file_path in selected_files:
        # 读取 CSV 文件
        df = pd.read_csv(file_path)
        
        # 获取文件名（不含路径和后缀）
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 提取第二列数据
        y_values = df.iloc[:, 1]
        x_values = range(1, len(y_values) + 1)
        
        # 使用 matplotlib 绘图
        plt.figure()
        plt.plot(x_values, y_values, label=file_name)
        plt.xlabel('trajectory')
        plt.ylabel('Plaquette')
        plt.title('Matplotlib')
        plt.ylim(0, 1)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(plt)
        
        # 使用 plotly 绘图
        fig = px.line(x=x_values, y=y_values, labels={'x': 'trajectory', 'y': 'Plaquette'}, title='Plotly')
        fig.update_layout(
            legend_title_text=file_name,
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig)

# 示例调用
# plot_plaquette(['data/file1.csv', 'data/file2.csv'])
if __name__ == "__main__":
    import os
    cur_abs_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_abs_path)
    data_dir = os.path.join(parent_dir, 'data')
    files = os.listdir(data_dir)
    file_paths = [os.path.join(data_dir, file) for file in files]
    plot_plaquette(file_paths)

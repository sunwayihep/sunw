import pandas as pd
import plotly.express as px
import streamlit as st
import os
from plotly import graph_objects as go

def get_su_folders(data_dir):
    # 获取所有以'su'开头的文件夹
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.startswith('su')]
    # 提取数字并排序
    folders.sort(key=lambda x: int(x[2:]) if x[2:].isdigit() else 0)
    return folders

def plot_plaquette_advanced():
    # 获取数据目录
    cur_abs_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_abs_path)
    data_dir = os.path.join(parent_dir, 'data')
    
    # 获取可选的SU文件夹
    su_folders = get_su_folders(data_dir)
    
    if not su_folders:
        st.error("在data目录下没有找到su开头的文件夹")
        return
    
    # 选择文件夹
    selected_folder = st.selectbox('选择SU文件夹', su_folders)
    folder_path = os.path.join(data_dir, selected_folder)
    
    # 获取所有CSV文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    file_paths = [os.path.join(folder_path, f) for f in files]
    
    # 创建一个空的图表容器用于所有曲线
    overview_plot = st.empty()
    
    # 绘制文件夹内所有曲线的总览图
    fig_all = go.Figure()
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        y_values = df.iloc[:, 1].tolist()
        x_values = list(range(1, len(y_values) + 1))
        
        fig_all.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            name=file_name,
            mode='lines'
        ))
    
    fig_all.update_layout(
        title=f'{selected_folder} 所有曲线总览',
        xaxis_title='trajectory',
        yaxis_title='Plaquette',
        yaxis_range=[0, 1]
    )
    overview_plot.plotly_chart(fig_all)
    
    # 创建多选框用于选择特定文件
    selected_files = st.multiselect('选择要显示的文件', files)
    
    # 创建一个空的图表容器用于选定的曲线
    selected_plot = st.empty()
    
    if selected_files:
        fig_selected = go.Figure()
        for file in selected_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            file_name = os.path.splitext(file)[0]
            y_values = df.iloc[:, 1].tolist()
            x_values = list(range(1, len(y_values) + 1))
            
            fig_selected.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                name=file_name,
                mode='lines'
            ))
        
        fig_selected.update_layout(
            title='选定文件的曲线',
            xaxis_title='trajectory',
            yaxis_title='Plaquette',
            yaxis_range=[0, 1]
        )
        selected_plot.plotly_chart(fig_selected)

if __name__ == "__main__":
    st.set_page_config(page_title="Plaquette Analysis", layout="wide")
    plot_plaquette_advanced() 
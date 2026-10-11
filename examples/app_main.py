import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from mechacode.solve import LegRestOptimizationProblem
from mechacode.mechanism import LegRestMechanism
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from mechacode.animation import create_mechanism_animation
from mechacode.solve import optimize_legrest

st.set_page_config(
    page_title="Legrest mechanism optimization",
    layout="wide",
)

st.title("Legrest mechanism optimization")

# ============================================================
# 侧边栏：页面切换
# ============================================================

page = st.sidebar.radio(
    "选择功能",
    ["🚀 Optimization", "📊 Visualization"],
    index=0,
)

# ============================================================
# 工具函数
# ============================================================

def save_result_to_json(result, problem, filename='data/result.json'):
    """保存 pymoo 结果到 JSON"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    data = {
        'metadata': {
            'algorithm': 'NSGA-II',
            'n_solutions': len(result.X),
            'n_var': problem.n_var,
            'n_obj': problem.n_obj,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'X': result.X.tolist(),
        'F': result.F.tolist(),
    }
    
    if result.G is not None:
        data['G'] = result.G.tolist()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filename


def load_result_from_json(filename='data/result.json'):
    """从 JSON 加载结果"""
    if not os.path.exists(filename):
        return None, None, None
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    X = np.array(data['X'])
    F = np.array(data['F'])
    
    return X, F, data


def create_pareto_chart(F, selected_idx=None):
    """创建帕累托前沿图"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=F[:, 0],
        y=F[:, 1],
        mode='markers',
        marker=dict(
            size=12,
            color='#3498db',
            opacity=0.7,
            line=dict(color='white', width=1),
        ),
        text=[f'解 {i}<br>目标1: {F[i,0]:.2f}<br>目标2: {F[i,1]:.2f}' 
              for i in range(len(F))],
        hoverinfo='text',
        name='帕累托解',
    ))
    
    # 高亮选中的点
    if selected_idx is not None:
        fig.add_trace(go.Scatter(
            x=[F[selected_idx, 0]],
            y=[F[selected_idx, 1]],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name=f'选中解 {selected_idx}',
        ))
    
    fig.update_layout(
        xaxis_title='包络面积 (最小化)',
        yaxis_title='位置误差 (最小化)',
        width=600,
        height=500,
        plot_bgcolor='white',
        hovermode='closest',
        dragmode='select',
    )
    
    return fig


# ============================================================
# 页面1：优化
# ============================================================

if page == "🚀 Optimization":
    st.header("🚀 Run optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ 参数设置")
        
        pop_size = st.slider("种群大小", 50, 300, 100, 10)
        n_gen = st.slider("迭代代数", 50, 500, 200, 10)
        seed = st.number_input("随机种子", value=42)
        min_distance = st.slider("最小间距", 5.0, 30.0, 15.0, 1.0)

        
        st.divider()
        
        # ✅ 优化按钮
        if st.button("🚀 开始优化", type="primary", use_container_width=True):
            with st.spinner("优化中，请稍候..."):
                # 创建问题
                problem = LegRestOptimizationProblem()
                problem.min_distance = min_distance

                # 创建问题实例
                problem = LegRestOptimizationProblem()
            
                # 生成初始种群
                initial_pop = problem.create_initial_population(pop_size=100)
            
                # 验证所有个体
                pop_result = problem.verify_population(initial_pop)

                #使用NSGA，在初始种群的情况下进行优化求解
                result, problem = optimize_legrest(pop_size=100, n_gen=200, seed=1)
                
                
                # ✅ 保存到 JSON
                filename = save_result_to_json(result, problem)
                
                # ✅ 保存到 session_state（用于页面间传递）
                st.session_state['optimization_result'] = {
                    'X': result.X.tolist(),
                    'F': result.F.tolist(),
                    'filename': filename,
                }
            
            st.success(f"✅ 优化完成！找到 {len(result.X)} 个帕累托最优解")
            st.info(f"📁 结果已保存到 `{filename}`")
    
    with col2:
        st.subheader("📊 优化结果")
        
        if 'optimization_result' in st.session_state:
            res = st.session_state['optimization_result']
            X = np.array(res['X'])
            F = np.array(res['F'])
            
            # 显示帕累托前沿
            fig = create_pareto_chart(F)
            st.plotly_chart(fig, use_container_width=True)
            
            # 统计信息
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("解的数量", len(F))
            with col_b:
                st.metric("包络面积范围", 
                         f"[{F[:,0].min():.1f}, {F[:,0].max():.1f}]")
            with col_c:
                st.metric("位置误差范围",
                         f"[{F[:,1].min():.1f}, {F[:,1].max():.1f}]")
            
            # ✅ 下载按钮
            st.divider()
            st.subheader("💾 导出")
            
            # 读取 JSON 文件内容
            with open(res['filename'], 'r', encoding='utf-8') as f:
                json_str = f.read()
            
            st.download_button(
                label="📥 下载 JSON 文件",
                data=json_str,
                file_name="result.json",
                mime="application/json",
                use_container_width=True,
            )
            
            # 导出 CSV
            df = pd.DataFrame(F, columns=['包络面积', '位置误差'])
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 下载 CSV",
                data=csv,
                file_name="result.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("点击左侧「开始优化」按钮运行优化")


# ============================================================
# 页面2：可视化
# ============================================================

elif page == "📊 Visualization":
    st.header("📊 结果可视化")
    
    # ✅ 数据源选择
    st.sidebar.header("📂 数据源")
    
    data_source = st.sidebar.radio(
        "选择数据来源",
        ["📁 本地文件", "📤 上传文件"],
    )
    
    X, F, data = None, None, None
    
    if data_source == "📁 本地文件":
        default_path = 'data/result.json'
        if os.path.exists(default_path):
            X, F, data = load_result_from_json(default_path)
            st.sidebar.success(f"✅ 已加载 `{default_path}`")
        else:
            st.sidebar.error(f"❌ 找不到 `{default_path}`")
            st.info("请先运行优化，或上传 JSON 文件")
            #st.stop()
    
    else:  # 上传文件
        uploaded = st.sidebar.file_uploader("上传 JSON", type=['json'])
        if uploaded:
            data = json.load(uploaded)
            X = np.array(data['X'])
            F = np.array(data['F'])
            st.sidebar.success("✅ 已加载上传的文件")
        else:
            st.info("请上传 JSON 文件")
            #st.stop()
    
    # 显示元数据
    if 'metadata' in data:
        st.sidebar.divider()
        st.sidebar.header("📋 元数据")
        for key, value in data['metadata'].items():
            st.sidebar.write(f"**{key}:** {value}")
    
    # ============================================================
    # 左右布局
    # ============================================================
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📈 帕累托前沿")
        st.caption("点击图中的点，查看对应的机构运动")
        
        # 创建帕累托图
        fig_pareto = create_pareto_chart(F)
        
        # ✅ 捕获点击事件
        event = st.plotly_chart(
            fig_pareto,
            use_container_width=True,
            on_select="rerun",
            key="pareto_chart",
        )
        
        # 获取选中的索引
        if event and event.selection and event.selection.points:
            selected_idx = event.selection.points[0].get('point_index', 0)
            st.success(f"✅ 选中解: {selected_idx}")
        else:
            selected_idx = 0
            st.info("👆 点击图中的点以选择解")
    
    with col_right:
        st.subheader("🎬 机构运动")
        
        # 显示选中解的信息
        st.write(f"**解 {selected_idx}**")
        st.write(f"包络面积: {F[selected_idx, 0]:.3f}")
        st.write(f"位置误差: {F[selected_idx, 1]:.3f}")
        
        # 生成机构动画
        points = X[selected_idx].reshape(7, 2)

        mechanism = LegRestMechanism(points)
        fig_anim = create_mechanism_animation(
                mechanism=mechanism,
                #angle_range=(angle_start, angle_end),
                angle_range=(np.pi/2 - mechanism.link_angles[3], np.pi/2),
        )
        
        st.plotly_chart(fig_anim, use_container_width=True)
    
    # ============================================================
    # 底部：数据表格
    # ============================================================
    
    st.divider()
    st.subheader("📋 所有解")
    
    df = pd.DataFrame(F, columns=['包络面积', '位置误差'])
    df['加权和'] = df.sum(axis=1)
    df = df.sort_values('加权和').reset_index(drop=True)
    
    st.dataframe(df.head(20), use_container_width=True)
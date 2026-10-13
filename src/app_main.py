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
    "Function choose: ",
    ["🚀 Optimization", "📊 Visualization"],
    index=0,
)

# ============================================================
# 工具函数
# ============================================================

def save_result_to_json(result, problem, target_lines, filename='data/result.json'):
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
        "target_lines": target_lines.tolist(),
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
        return None, None, None, None
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    X = np.array(data['X'])
    F = np.array(data['F'])
    target_lines = np.array(data["target_lines"])
    
    return X, F, target_lines, data


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
        st.subheader("参数设置")
        
        pop_size = st.slider("种群大小", 50, 300, 100, 10)
        n_gen = st.slider("迭代代数", 50, 500, 200, 10)
        seed = st.number_input("随机种子", value=1)
        min_distance = st.slider("最小间距", 5.0, 30.0, 15.0, 1.0)

        
        st.divider()

        st.subheader("输入需要拟合的直线：")
        # 初始数据（对应你当前的 target_lines）
        default_df = pd.DataFrame({
            "Line": ["L1", "L2"],
            "start_X": [168, 193],
            "start_Y": [-4, -40],
            "end_X": [222, 260],
            "end_Y": [69, 20]
        })

        edited_df = st.data_editor(
            default_df,
            num_rows="dynamic",  # 固定 2 行；要允许增删改为 "dynamic"
            width='content',
            key="target_lines_editor"
        )

        # 转换为 target_lines 格式
        target_lines = np.array([
            [[row["start_X"], row["start_Y"]], [row["end_X"], row["end_Y"]]]
            for _, row in edited_df.iterrows()
        ])

        #print(target_lines.tolist())

        st.divider()
        
        st.subheader("输入初始连杆的布置范围大小(X最大,Y最大):")
        # 初始数据（对应你当前的 area_size）
        default_area_df = pd.DataFrame({
            "Area": ["X", "Y"],
            "Value": [70, 240],
        })

        edited_area_df = st.data_editor(
            default_area_df,
            num_rows="fixed",  # 固定 数量
            width='content',
            key="area_size_editor"
        )

        # 转换为 area_size 格式
        area_size = edited_area_df["Value"].tolist()
        
        st.divider()

        # ✅ 优化按钮
        if st.button("🚀 开始优化", type="primary", width='stretch'):
            with st.spinner("优化中，请稍候..."):
                # 创建问题实例
                #需要拟合的直线段
                #target_lines = np.array([[[168, -4], [221.67, 68.48]], [[193.5, -40.5], [260.1, 20.3]]])
                problem = LegRestOptimizationProblem(target_lines, min_distance, area_size)
            
                # 生成初始种群

                initial_pop = problem.create_initial_population(pop_size)
            
                # 验证所有个体
                pop_result = problem.verify_population(initial_pop)

                print("=" * 50)
                print("种群验证结果")
                print("=" * 50)
                print(f"总个体数: {pop_result['total']}")
                print(f"有效个体数: {pop_result['valid']}")
                print(f"无效个体数: {pop_result['invalid']}")
                print(f"有效比例: {pop_result['valid_ratio']:.2f}%")
                if pop_result['invalid'] > 0:
                    print(f"\n无效个体索引: {pop_result['invalid_indices']}")
                else:
                    print("\n✅ 所有个体都满足约束！")

                #使用NSGA，在初始种群的情况下进行优化求解
                result= optimize_legrest(problem, initial_pop, pop_size, n_gen, seed )
                
                
                # ✅ 保存到 JSON
                filename = save_result_to_json(result, problem, target_lines)
                
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
            fig.update_layout(
                    dragmode='pan',
                    paper_bgcolor='rgba(0,0,0,0)',   # 整个图表的背景（含标题、图例区域）
                    plot_bgcolor='rgba(0,0,0,0)'     # 绘图区（坐标轴内部）的背景
                )
            st.plotly_chart(fig, width='stretch')
            
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
                width='stretch',
            )
            
            # 导出 CSV
            df = pd.DataFrame(F, columns=['包络面积', '位置误差'])
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 下载 CSV",
                data=csv,
                file_name="result.csv",
                mime="text/csv",
                width='stretch',
            )
        else:
            st.info("点击左侧「开始优化」按钮运行优化")


# ============================================================
# 页面2：可视化
# ============================================================

elif page == "📊 Visualization":
    st.header("📊 结果可视化")
        
    X, F, data = None, None, None
    with st.sidebar.container(border=True):
        st.markdown("**📂 数据源**")
        data_source = st.radio(
            "选择数据来源",
            ["📁 本地文件", "📤 上传文件"],
            label_visibility="collapsed",
        )
        if data_source == "📁 本地文件":
            default_path = 'data/result.json'
            if os.path.exists(default_path):
                X, F, target_lines, data = load_result_from_json(default_path)
                st.sidebar.success(f"✅ 已加载 `{default_path}`")
            else:
                st.sidebar.error(f"❌ 找不到 `{default_path}`")
                st.info("请先运行优化，或上传 JSON 文件")
                st.stop()
        
        else:  # 上传文件
            with st.form("upload_form", clear_on_submit=True):
                uploaded = st.sidebar.file_uploader("上传 JSON 文件", type=['json'], accept_multiple_files=False)
                
                if uploaded:
                    data = json.load(uploaded)
                    X = np.array(data['X'])
                    F = np.array(data['F'])
                    target_lines = np.array(data['target_lines'])
                    st.sidebar.success("✅ 已加载上传的文件")
                    
                else:
                    st.info("请上传 JSON 文件")
                    st.stop()
        #target_lines = np.array([[[168, -4], [221.67, 68.48]], [[193.5, -40.5], [260.1, 20.3]]])
        # 显示元数据
        if 'metadata' in data:
            st.sidebar.divider()
            st.sidebar.header("📋 元数据")
            for key, value in data['metadata'].items():
                st.sidebar.write(f"**{key}:** {value}")

        st.sidebar.divider()

        #显示目标线点
        #将array转换成pandas
        st.sidebar.write("🌀 Target Lines")
        target_lines_xy = target_lines.reshape(target_lines.shape[0], -1)
        target_lines_df = pd.DataFrame(target_lines_xy, columns=["x1", "y1", "x2", "y2"])
        st.sidebar.dataframe(target_lines_df, width="content")
    
    # ============================================================
    # 上下布局
    # ============================================================
    
    
    st.subheader("📈 帕累托前沿")
    st.caption("点击图中的点，查看对应的机构运动")

    
    # 创建帕累托图
    fig_pareto = create_pareto_chart(F)
    fig_pareto.update_layout(
        dragmode='pan',
        paper_bgcolor='rgba(0,0,0,0)',   # 整个图表的背景（含标题、图例区域）
        plot_bgcolor='rgba(0,0,0,0)'     # 绘图区（坐标轴内部）的背景
    )
        
    
    # ✅ 捕获点击事件
    event = st.plotly_chart(
        fig_pareto,
        width='stretch',
        on_select="rerun",
        key="pareto_chart",
    )


    
    # 获取选中的索引
    if event and event.selection and event.selection.points:
        selected_idx = event.selection.points[0].get('point_index', 0)
        st.success(f"✅ 选中解: {selected_idx}")
    else:
        selected_idx = 0
        st.info("👆 框选图中的点以选择解")



      

    #st.divider()
    
    # ----- 下面：机构动画 -----
    st.subheader("🎬 机构运动")
    
    # 显示选中解的信息
    #st.write(f"**solution {selected_idx}** \t 包络面积: {F[selected_idx, 0]:.3f} \t位置误差: {F[selected_idx, 1]:.3f}")
    st.markdown(
    f"| solution | 包络面积 | 位置误差 |\n"
    f"|---|---|---|\n"
    f"| {selected_idx} | {F[selected_idx, 0]:.3f} | {F[selected_idx, 1]:.3f} |"
)
    
    
    # 生成机构动画
    points = X[selected_idx].reshape(7, 2)

    mechanism = LegRestMechanism(points)
    fig_anim = create_mechanism_animation(
            mechanism=mechanism,
            target_lines = target_lines,
            angle_range=(np.pi/2 - mechanism.link_angles[3], np.pi/2),
    )

    fig_anim.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',   # 整个图表的背景（含标题、图例区域）
            plot_bgcolor='rgba(0,0,0,0)'     # 绘图区（坐标轴内部）的背景
        )
    
    st.plotly_chart(fig_anim, width='stretch')
    
    # ============================================================
    # 底部：数据表格
    # ============================================================
    
    st.divider()
    st.subheader("📋 该解的各点位置")
    
    df = pd.DataFrame(points, columns=['X', 'Y'], index=['P1','P2','P3','P4','P5','P6','P7'])
    
    st.dataframe(df.head(20), width='content')
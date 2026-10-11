import streamlit as st
import numpy as np
import plotly.graph_objects as go

from mechacode.mechanism import LegRestMechanism
from mechacode.animation import create_mechanism_animation

# ===== 页面配置 =====
st.set_page_config(
    page_title="腿托机构优化",
    page_icon="🦵",
    layout="wide",
)

st.title("🦵 腿托机构多目标优化")

# ===== 侧边栏：参数输入 =====
st.sidebar.header("⚙️ 机构参数")

# 机构坐标输入
st.sidebar.subheader("7个点坐标")
default_points = np.array([
            [
                36.539245982427595,
                28.024617410140323
            ],
            [
                35.87942545834548,
                61.612532611381255
            ],
            [
                36.30424776629044,
                111.28006691119293
            ],
            [
                38.32926341115385,
                93.40580470804511
            ],
            [
                43.39362579184341,
                26.051947111474092
            ],
            [
                42.43135186300394,
                113.54397036595957
            ],
            [
                43.96935685036926,
                62.258478880845594
            ]
        ],
)

points_input = []
for i in range(7):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x = st.number_input(f"P{i+1} x", value=float(default_points[i, 0]),
                           key=f"x{i}")
    with col2:
        y = st.number_input(f"P{i+1} y", value=float(default_points[i, 1]),
                           key=f"y{i}")
    points_input.append([x, y])

points = np.array(points_input)

# 动画参数
st.sidebar.header("🎬 动画参数")
#angle_start = st.sidebar.slider("起始角度 (rad)", -1.0, 0.0, -0.5, 0.1)
#angle_end = st.sidebar.slider("结束角度 (rad)", 0.0, 1.0, 0.5, 0.1)
n_frames = st.sidebar.slider("帧数", 10, 100, 50, 5)
duration = st.sidebar.slider("帧间隔 (ms)", 10, 200, 50, 10)

# ===== 主区域 =====
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎬 机构运动动画")

    # 创建机构
    mechanism = LegRestMechanism(points)
    
    # 创建动画
    fig = create_mechanism_animation(
        mechanism=mechanism,
        #angle_range=(angle_start, angle_end),
        angle_range=(np.pi/2 - mechanism.link_angles[3], np.pi/2),
        n_frames=n_frames,
        duration=duration,
    )
    
    # 显示
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 机构信息")
    
    st.write("**连杆连接关系:**")
    for k, (i, j) in enumerate(mechanism.link_definitions):
        label = "🔴 驱动杆" if k == mechanism.driver_link_idx else "🔵 普通杆"
        st.write(f"  {label}: P{i+1} - P{j+1}")
    
    st.write("**关键点:**")
    st.write(f"  固定点: P{mechanism.fixed_point_idx[0] + 1}, P{mechanism.fixed_point_idx[1] + 1}")
    st.write(f"  驱动杆: P{mechanism.link_definitions[mechanism.driver_link_idx-1][0] + 1} - "
             f"P{mechanism.link_definitions[mechanism.driver_link_idx-1][1] + 1}")
    st.write(f"  末端执行器: P{mechanism.end_effector_idx[0] + 1} - P{mechanism.end_effector_idx[1] + 1}")
    
    # 计算连杆长度
    st.write("**连杆长度:**")
    for k, (i, j) in enumerate(mechanism.link_definitions):
        length = np.linalg.norm(points[j] - points[i])
        st.write(f"  杆 {k+1} (P{i+1}-P{j+1}): {length:.2f} mm")

# ===== 底部：数据下载 =====
st.divider()
st.subheader("💾 导出")

col_a, col_b, col_c = st.columns(3)

with col_a:
    # 保存 HTML
    html_str = fig.to_html(include_plotlyjs='cdn')
    st.download_button(
        label="📥 下载动画 HTML",
        data=html_str,
        file_name="mechanism_animation.html",
        mime="text/html",
    )

with col_b:
    # 保存坐标
    points_csv = "\n".join([f"P{i+1},{p[0]},{p[1]}" 
                           for i, p in enumerate(points)])
    points_csv = "name,x,y\n" + points_csv
    st.download_button(
        label="📥 下载坐标 CSV",
        data=points_csv,
        file_name="mechanism_points.csv",
        mime="text/csv",
    )

with col_c:
    # 保存图片
    st.write("💡 提示：右键动画图可保存为 PNG")
import plotly.graph_objects as go
import numpy as np


def create_plotly_animation(mechanism, angle_range):
    """创建 Plotly 动画"""
    driver_angles, joints = mechanism.forward_kinematics_interval(
        (angle_range[0], angle_range[1]), np.deg2rad(2)
    )
    
    n_frames = joints.shape[0]
    
    # 创建帧
    frames = []
    for frame in range(n_frames):
        points = joints[frame]
        
        # 每帧的数据
        frame_data = []
        
        # 连杆
        for i, j in mechanism.link_definitions:
            frame_data.append(
                go.Scatter(
                    x=[points[i, 0], points[j, 0]],
                    y=[points[i, 1], points[j, 1]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False,
                )
            )
        
        # 节点
        frame_data.append(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode='markers+text',
                marker=dict(size=10, color='red'),
                text=[f'P{i+1}' for i in range(7)],
                textposition='top center',
                showlegend=False,
            )
        )
        
        frames.append(go.Frame(data=frame_data, name=str(frame)))
    
    # 初始帧
    initial_points = joints[0]
    initial_data = []
    for i, j in mechanism.link_definitions:
        initial_data.append(
            go.Scatter(
                x=[initial_points[i, 0], initial_points[j, 0]],
                y=[initial_points[i, 1], initial_points[j, 1]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False,
            )
        )
    initial_data.append(
        go.Scatter(
            x=initial_points[:, 0],
            y=initial_points[:, 1],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=[f'P{i+1}' for i in range(7)],
            textposition='top center',
            showlegend=False,
        )
    )
    
    # 创建图形
    fig = go.Figure(
        data=initial_data,
        frames=frames,
        layout=go.Layout(
            title='腿托机构动画',
            xaxis=dict(title='X', autorange='reversed'),
            yaxis=dict(title='Y', autorange='reversed', scaleanchor='x'),
            updatemenus=[dict(
                type='buttons',
                buttons=[
                    dict(label='▶ 播放', method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True)]),
                    dict(label='⏸ 暂停', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate')]),
                ],
            )],
            sliders=[dict(
                steps=[dict(method='animate',
                           args=[[str(f)], dict(mode='immediate',
                                                frame=dict(duration=50, redraw=True),
                                                transition=dict(duration=0))],
                           label=str(f)) for f in range(n_frames)],
            )],
        )
    )
    
    return fig


# Streamlit 中使用
fig = create_plotly_animation(mechanism, (-0.5, 0.5))
st.plotly_chart(fig, use_container_width=True)
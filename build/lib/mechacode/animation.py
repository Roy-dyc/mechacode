import numpy as np
import plotly.graph_objects as go


def create_mechanism_animation(mechanism, angle_range=(0, np.pi/2),
                                n_frames=50, duration=50):
    """
    创建机构的 Plotly 动画
    
    参数:
    mechanism: LegRestMechanism 实例
    angle_range: (起始角度, 结束角度)
    n_frames: 帧数
    duration: 每帧持续时间（毫秒）
    """
    # 获取运动学数据
    driver_angles, joints = mechanism.forward_kinematics_interval(
        angle_range, (angle_range[1] - angle_range[0]) / n_frames
    )
    
    n_frames_actual = joints.shape[0]
    
    # ===== 1. 计算坐标范围 =====
    x_min = joints[:, :, 0].min() - 10
    x_max = joints[:, :, 0].max() + 10
    y_min = joints[:, :, 1].min() - 10
    y_max = joints[:, :, 1].max() + 10
    
    # ===== 2. 创建帧 =====
    frames = []
    for frame_idx in range(n_frames_actual):
        frame_points = joints[frame_idx]
        frame_data = []
        
        # 普通连杆
        for k, (i, j) in enumerate(mechanism.link_definitions):
            if k == mechanism.driver_link_idx:
                continue
            frame_data.append(
                go.Scatter(
                    x=[frame_points[i, 0], frame_points[j, 0]],
                    y=[frame_points[i, 1], frame_points[j, 1]],
                    mode='lines',
                    line=dict(color='#3498db', width=3),
                    showlegend=False,
                    hoverinfo='skip',
                )
            )
        
        # 驱动杆
        i, j = mechanism.link_definitions[mechanism.driver_link_idx]
        frame_data.append(
            go.Scatter(
                x=[frame_points[i, 0], frame_points[j, 0]],
                y=[frame_points[i, 1], frame_points[j, 1]],
                mode='lines',
                line=dict(color='#e74c3c', width=5),
                showlegend=False,
                hoverinfo='skip',
            )
        )
        
        # 节点
        frame_data.append(
            go.Scatter(
                x=frame_points[:, 0],
                y=frame_points[:, 1],
                mode='markers+text',
                marker=dict(size=12, color='#2c3e50',
                           line=dict(color='white', width=2)),
                text=[f'P{i+1}' for i in range(len(frame_points))],
                textposition='top center',
                textfont=dict(size=11, color='black'),
                showlegend=False,
                hoverinfo='text',
            )
        )
        
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
    
    # ===== 3. 初始帧 =====
    initial_points = joints[0]
    initial_data = []
    
    for k, (i, j) in enumerate(mechanism.link_definitions):
        if k == mechanism.driver_link_idx:
            continue
        initial_data.append(
            go.Scatter(
                x=[initial_points[i, 0], initial_points[j, 0]],
                y=[initial_points[i, 1], initial_points[j, 1]],
                mode='lines',
                line=dict(color='#3498db', width=3),
                showlegend=False,
                hoverinfo='skip',
            )
        )
    
    i, j = mechanism.link_definitions[mechanism.driver_link_idx]
    initial_data.append(
        go.Scatter(
            x=[initial_points[i, 0], initial_points[j, 0]],
            y=[initial_points[i, 1], initial_points[j, 1]],
            mode='lines',
            line=dict(color='#e74c3c', width=5),
            showlegend=False,
            hoverinfo='skip',
        )
    )
    
    initial_data.append(
        go.Scatter(
            x=initial_points[:, 0],
            y=initial_points[:, 1],
            mode='markers+text',
            marker=dict(size=12, color='#2c3e50',
                       line=dict(color='white', width=2)),
            text=[f'P{i+1}' for i in range(len(initial_points))],
            textposition='top center',
            textfont=dict(size=11, color='black'),
            showlegend=False,
            hoverinfo='text',
        )
    )
    
    # ===== 4. 创建图形 =====
    fig = go.Figure(
        data=initial_data,
        frames=frames,
        layout=go.Layout(
            title=dict(text='腿托机构运动动画', font=dict(size=20), x=0.5),
            xaxis=dict(
                title='X (mm)',
                side = "top",
                range=[x_max, x_min],
                showgrid=True,
                gridcolor='#eee',
                zeroline=False,
                autorange=False,  
            ),
            yaxis=dict(
                title='Y (mm)',
                side = "right",
                range=[y_max, y_min],
                showgrid=True,
                gridcolor='#eee',
                zeroline=False,
                scaleanchor='x',
                scaleratio=1,
                autorange=False,  
            ),
            width=800,
            height=800,
            plot_bgcolor='white',
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    x=0.05,
                    y=1.12,
                    xanchor='left',
                    buttons=[
                        dict(
                            label='▶ 播放',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=duration, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            )],
                        ),
                        dict(
                            label='⏸ 暂停',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate',
                            )],
                        ),
                    ],
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            method='animate',
                            args=[[str(f)], dict(
                                mode='immediate',
                                frame=dict(duration=duration, redraw=True),
                                transition=dict(duration=0),
                            )],
                            label=str(f),
                        )
                        for f in range(n_frames_actual)
                    ],
                    x=0.05,
                    y=-0.05,
                    len=0.9,
                    xanchor='left',
                    yanchor='top',
                    currentvalue=dict(
                        prefix='帧: ',
                        visible=True,
                        xanchor='right',
                    ),
                ),
            ],
        ),
    )
    
    return fig
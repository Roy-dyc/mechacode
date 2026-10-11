import numpy as np


class LegRestMechanism:
    """腿托机构"""
    
    def __init__(self, points):
        self.points_initial = np.array(points)
        self.points_current = self.points_initial.copy()
        
        # 连杆定义
        self.link_definitions = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            (0, 3), (1, 4), (2, 5), (3, 6), (0, 6),
        ]
        
        # 驱动杆索引
        self.driver_link_idx = 6  # P1-P4
        
        # 末端执行器
        self.end_effector_idx = 4  # P5
        
        # 固定点
        self.fixed_point_idx = 0  # P1
    
    def forward_kinematics(self, driver_angle):
        """
        正向运动学：给定驱动角度，返回所有点坐标
        这里用简化模型演示，实际项目替换为真实运动学
        """
        points = self.points_initial.copy()
        
        # 驱动杆 P1-P4
        i, j = self.link_definitions[self.driver_link_idx]
        dx = self.points_initial[j, 0] - self.points_initial[i, 0]
        dy = self.points_initial[j, 1] - self.points_initial[i, 1]
        L = np.sqrt(dx**2 + dy**2)
        
        # 更新 P4 位置
        points[j, 0] = self.points_initial[i, 0] + L * np.cos(driver_angle)
        points[j, 1] = self.points_initial[i, 1] + L * np.sin(driver_angle)
        
        # 简化：其他点做平移
        delta = points[j] - self.points_initial[j]
        for k in range(len(points)):
            if k != i and k != j:
                points[k] = self.points_initial[k] + delta * 0.5
        
        return points
    
    def forward_kinematics_interval(self, angle_range, step):
        """在角度区间内生成多帧"""
        start_angle, end_angle = angle_range
        
        # 计算初始驱动杆角度
        i, j = self.link_definitions[self.driver_link_idx]
        dx = self.points_initial[j, 0] - self.points_initial[i, 0]
        dy = self.points_initial[j, 1] - self.points_initial[i, 1]
        initial_angle = np.arctan2(dy, dx)
        
        # 生成角度序列
        angles = np.arange(initial_angle + start_angle,
                          initial_angle + end_angle,
                          step)
        
        # 计算每帧
        joints = []
        for angle in angles:
            points = self.forward_kinematics(angle)
            joints.append(points)
        
        return angles, np.array(joints)
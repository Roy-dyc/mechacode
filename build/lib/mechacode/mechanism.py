# this file include the class which could represent the mechanism

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict
from scipy.optimize import fsolve
from operator import itemgetter


class LegRestMechanism:
    """
    基于复数极坐标法的腿托多连杆机构运动学求解器
    使用7个关键点定义机构拓扑
    """
    
    def __init__(self, points: np.ndarray):
        """
        初始化机构
        
        参数:
        points: 7x2的数组,包含7个点的(x,y)坐标
               [[x1,y1], [x2,y2], ..., [x7,y7]]
        """
        self.points_initial = points.copy()  # 保存初始坐标
        self.points_current = points.copy()  # 当前坐标
        
        # 定义连杆连接关系 (端点索引，0-based)
        self.link_definitions = [
            (0, 1),  # 连杆1: 1-2
            (1, 2),  # 连杆2: 2-3
            (2, 3),  # 连杆3: 3-4
            (0, 3),  # 连杆4: 1-4(驱动杆)
            (3, 4),  # 连杆5: 4-5
            (3, 5),  # 连杆6: 4-6
            (5, 6),  # 连杆7: 6-7 
            (4, 6),  # 连杆8: 5-7
        ]
        
        # 计算连杆参数
        self.link_lengths = self._calculate_link_lengths()
        self.link_angles = self._calculate_link_angles()
        
        # 识别驱动杆 (假设连杆4，即1-4为驱动杆)
        self.driver_link_idx = 4  # 0-based索引
        
        # 识别固定点组 (杆1为固定杆，即点1，2为固定点)
        self.fixed_point_idx = (0, 1)

        #识别该机构中固定的连杆
        self.fixed_pair_links = [
            (0, 3, 5), #连杆 1-4-6
            (4, 3, 2), #连杆 5-4-3
        ]

        #计算两组固定连杆的角度

        self.fixed_pair_links_angle = self._calculate_2links_angle()
        
        # 识别末端执行器组 (杆8为末端，即点（5，7）为末端点组)
        self.end_effector_idx = (4, 6)
        
        # # 验证机构
        # self._validate_mechanism()
    
    def _calculate_link_lengths(self) -> Dict[int, float]:
        """
        计算所有连杆的长度
        """
        lengths = {}
        for idx, (i, j) in enumerate(self.link_definitions):
            dx = self.points_initial[j, 0] - self.points_initial[i, 0]
            dy = self.points_initial[j, 1] - self.points_initial[i, 1]
            lengths[idx] = np.sqrt(dx**2 + dy**2)
        return lengths
    
    def _calculate_link_angles(self) -> Dict[int, float]:
        """
        计算所有连杆的初始角度 (弧度) 与 x轴的角度
        使用复数极坐标法: angle = atan2(dy, dx)
        """
        angles = {}
        for idx, (i, j) in enumerate(self.link_definitions):
            dx = self.points_initial[j, 0] - self.points_initial[i, 0]
            dy = self.points_initial[j, 1] - self.points_initial[i, 1]
            angles[idx] = np.arctan2(dy, dx)
        return angles
    

    def _calculate_2links_angle(self) ->np.ndarray:
        '''
        计算两根连杆之间的角度, 得到该legrest机构中两个固定角度
        ba->bc, 顺时针转动角度为负； 逆时针转动角度为正
        '''
        same_links = np.array(self.fixed_pair_links)
        same_link_points_a =  self.points_initial[same_links[:,0]]
        same_link_points_b =  self.points_initial[same_links[:,1]]
        same_link_points_c =  self.points_initial[same_links[:,2]]

        # 向量化计算
        ba = same_link_points_a - same_link_points_b
        bc = same_link_points_c - same_link_points_b

        # 叉积和点积
        cross = ba[:, 0] * bc[:, 1] - ba[:, 1] * bc[:, 0]
        dot = ba[:, 0] * bc[:, 0] + ba[:, 1] * bc[:, 1]

        # 批量计算（利用 NumPy 的向量化）
        radians = np.arctan2(cross, dot)
        radians = np.where(radians>0, 2*np.pi-radians, -radians)
        

        return radians
            
    def _kinematic_equations(self, equ_vars, drive_angle, equ_pa):
        """
            该函数用于forward_kinematic函数的非线性方程组
            equ_vars：需要求解的角度，此结构为（theta2, theta3, theta5, theta6, theta7, theta8),六个变量
            drive_angle：驱动角定义为与y轴的夹角，也就是公式中需要np.pi/2 - drive_angle
            equ_pa：为参数(theta1, fix_angle_1, fix_angle_2, L1, L2, L3, L4, L5, L6, L7，L8)
            其中fix_angle_1(连杆4，6的夹角，顺时针角度), fix_angle_2(连杆5，3的夹角，顺时针角度), theta1 为固定角度
        """
        #获取需要的参数
        var_theta2, var_theta3, var_theta5, var_theta6, var_theta7, var_theta8 = equ_vars
        var_theta4 = np.pi/2 - drive_angle
        var_theta1, fix_angle_1, fix_angle_2, L1, L2, L3, L4, L5, L6, L7, L8 = equ_pa
        
        #列方程,六个未知数，六个方程
        equ1 = L1*np.cos(var_theta1) + L2*np.cos(var_theta2) + L3 * np.cos(var_theta3) - L4 * np.cos(var_theta4)
        equ2 = L1*np.sin(var_theta1) + L2*np.sin(var_theta2) + L3 * np.sin(var_theta3) - L4 * np.sin(var_theta4)
        equ3 = L6*np.cos(var_theta6) + L7*np.cos(var_theta7) - L5 * np.cos(var_theta5) - L8 * np.cos(var_theta8)
        equ4 = L6*np.sin(var_theta6) + L7*np.sin(var_theta7) - L5 * np.sin(var_theta5) - L8 * np.sin(var_theta8)
        equ5 = np.pi - fix_angle_1 + np.abs(var_theta4) - np.abs(var_theta6)
        equ6 = np.pi - fix_angle_2 + np.abs(var_theta3) - np.abs(var_theta5)

        return [equ1, equ2, equ3, equ4, equ5, equ6]
        
    
    def forward_kinematics_angle(self, driver_angle: float) -> np.ndarray:
        """
        该函数输入一个驱动角度，返回该驱动角度下的其他的连杆对应的角度
        正向运动学求解：给定驱动杆角度，计算所有点坐标
        使用复数极坐标法
        
        
        参数:
        driver_angle: 驱动杆(1-4)的绝对角度 (弧度)
        
        返回:
        points: 7x2的数组，所有点的坐标
        """
        #赋予L1-8 连杆长度  theta1-8 各连杆与x轴正方向的角度
        L1 = self.link_lengths[0]
        L2 = self.link_lengths[1]
        L3 = self.link_lengths[2]
        L4 = self.link_lengths[3]
        L5 = self.link_lengths[4]
        L6 = self.link_lengths[5]
        L7 = self.link_lengths[6]
        L8 = self.link_lengths[7]

        #机构初始角度
        theta1 = self.link_angles[0]
        theta2 = self.link_angles[1]
        theta3 = self.link_angles[2]
        theta4 = self.link_angles[3]
        theta5 = self.link_angles[4]
        theta6 = self.link_angles[5]
        theta7 = self.link_angles[6]
        theta8 = self.link_angles[7]

        #两个固定连杆的角度
        constant_angle_link4_6 = self.fixed_pair_links_angle[0]
        constant_angle_link5_3 = self.fixed_pair_links_angle[1]

        #通过_kinematic_equations(equ_vars, drive_angle, equ_pa)函数根据连杆关系及约束建立非线性方程组
        #对于不同角度可以使用在变量名前面加var与上述实际角度区分
        #对参数进行赋值
        driver_angle_value = driver_angle
        equ_pa = (theta1, constant_angle_link4_6, constant_angle_link5_3, L1, L2, L3, L4, L5, L6, L7, L8)
        #使用scipy的fsolve函数，需要给一个变量的初始值使用结构的初始值
        var_guess = (theta2, theta3, theta5, theta6, theta7, theta8)

        #有了参数及初值猜想值，进行求解
        angle_solutions, info, ier, mesg = fsolve(self._kinematic_equations, var_guess, args = (driver_angle_value, equ_pa), full_output=True)      

        # 关键：检查 fvec 是否都接近 0
        max_residual = np.max(np.abs(info['fvec']))
        print(f"最大残差: {max_residual:.2e}")

        
        # # 更新当前坐标
        # self.points_current = points.copy()
        if ier == 1 and max_residual < 1e-6:
            print("✅ 求解完美成功！")
            angle_solutions = np.insert(angle_solutions, [0, 2], [theta1, np.pi/2 - driver_angle_value])
            return angle_solutions

        elif ier == 1 and max_residual > 1e-6:
            print("⚠️ 表面收敛但残差较大，解可能不可靠")
            
        else:
            print(f"❌ 求解失败: {mesg}")

    def forward_kinematics_interval(self, driver_interval: tuple[float, float], step: float) -> np.ndarray:
        """
        该函数输入一个驱动角度区间，返回该驱动角度下的其他的连杆对应的角度
        正向运动学求解：给定驱动杆角度，计算所有点坐标
        使用复数极坐标法
        
        参数:
        driver_interval: 驱动杆(1-4)的绝对角度，弧度

        
        
        返回:
        7x2的数组，所有点的坐标
        """
        #赋予L1-8 连杆长度  theta1-8 各连杆与x轴正方向的角度
        L1 = self.link_lengths[0]
        L2 = self.link_lengths[1]
        L3 = self.link_lengths[2]
        L4 = self.link_lengths[3]
        L5 = self.link_lengths[4]
        L6 = self.link_lengths[5]
        L7 = self.link_lengths[6]
        L8 = self.link_lengths[7]

        #机构初始角度
        theta1 = self.link_angles[0]
        theta2 = self.link_angles[1]
        theta3 = self.link_angles[2]
        theta4 = self.link_angles[3]
        theta5 = self.link_angles[4]
        theta6 = self.link_angles[5]
        theta7 = self.link_angles[6]
        theta8 = self.link_angles[7]

        #两个固定连杆的角度
        constant_angle_link4_6 = self.fixed_pair_links_angle[0]
        constant_angle_link5_3 = self.fixed_pair_links_angle[1]

        #通过_kinematic_equations(equ_vars, drive_angle, equ_pa)函数根据连杆关系及约束建立非线性方程组
        #对于不同角度可以使用在变量名前面加var与上述实际角度区分
        #对参数进行赋值
        equ_pa = (theta1, constant_angle_link4_6, constant_angle_link5_3, L1, L2, L3, L4, L5, L6, L7, L8)
        #使用scipy的fsolve函数，需要给一个变量的初始值使用结构的初始值
        var_guess = (theta2, theta3, theta5, theta6, theta7, theta8)

        #下面通过按照step 循环遍历angle_interval,初始角度为layout角度也就是theta4, 最终的角度根据实际应用场景定义
        #获取驱动角度序列
        interval_angles = np.arange(driver_interval[0], driver_interval[1], step)
        theta_results = [] #该list存放获得结果 如果求解有问题就停止
        #优先把初始角度放入
        initial_theta = list(self.link_angles.values())
        theta_results.append(initial_theta)
        guess_flag = 0 #该flag存放guess的结果
        indices = [1,2, 4,5,6,7]
        angle_solutions = list(itemgetter(*indices)(theta_results[0]))
        #遍历计算,前一个计算结果为后一个计算结果的guess值
        for every_angle in interval_angles.tolist()[1:]:
            #每次的guess都是上一次求解的结果
            #var_guess = theta_results[guess_flag]
            #有了参数及猜想值，进行求解
            angle_solutions, info, ier, mesg = fsolve(self._kinematic_equations, angle_solutions, args = (every_angle, equ_pa), full_output=True)
            # 关键：检查 fvec 是否都接近 0
            max_residual = np.max(np.abs(info['fvec']))
            if ier == 1 and max_residual < 1e-6:
                angle_all = np.insert(angle_solutions, [0, 2], [theta1, np.pi/2 - every_angle]).tolist()
                #print(angle_all)
                theta_results.append(angle_all) #将结果放入theta_results
                guess_flag+=1 # guess_flag加1
            else:
                break; #找到一个不满足的, 根据实际情况后面都不能满足，所以跳出循环            
        #将求解的结果变为np.ndarry中
        theta_results = np.array(theta_results)

        #将theta_results 角度变成各转轴的坐标点
        step_number = theta_results.shape[0]

        p1x = np.tile(self.points_initial[0,0], (step_number,))
        p1y = np.tile(self.points_initial[0,1], (step_number,))

        p2x = p1x + np.cos(theta_results[:,0])*self.link_lengths[0]
        p2y = p1y + np.sin(theta_results[:,0])*self.link_lengths[0]

        p3x = p2x + np.cos(theta_results[:,1])*self.link_lengths[1]
        p3y = p2y + np.sin(theta_results[:,1])*self.link_lengths[1]

        p4x = p1x + np.cos(theta_results[:,3])*self.link_lengths[3]
        p4y = p1y + np.sin(theta_results[:,3])*self.link_lengths[3]

        p5x = p4x + np.cos(theta_results[:,4])*self.link_lengths[4]
        p5y = p4y + np.sin(theta_results[:,4])*self.link_lengths[4]

        p6x = p4x + np.cos(theta_results[:,5])*self.link_lengths[5]
        p6y = p4y + np.sin(theta_results[:,5])*self.link_lengths[5]

        p7x = p5x + np.cos(theta_results[:,7])*self.link_lengths[7]
        p7y = p5y + np.sin(theta_results[:,7])*self.link_lengths[7]

        X_results = np.stack([p1x, p2x, p3x, p4x, p5x, p6x, p7x], axis = 1)
        Y_results = np.stack([p1y, p2y, p3y, p4y, p5y, p6y, p7y], axis = 1)

        all_points = np.stack([X_results, Y_results], axis = 2)

        return interval_angles, all_points
        
        
        
    
    
    def animate_mechanism(self, angle_range: Tuple[float, float], 
                           interval: int = 50):
        """
        创建机构运动的动画
        
        参数:
        angle_range: 驱动杆的角度范围 (起始角度, 结束角度) 弧度
        num_frames: 帧数
        interval: 帧间隔 (毫秒)
        """
        #获取所有时刻的驱动点
        driver_angles, joints = self.forward_kinematics_interval((angle_range[0], angle_range[1]), np.deg2rad(1))
        # print(f"angles 长度: {len(driver_angles)}")
        # print(f"joints 长度: {len(joints)}")

        #获取画布的最大值最小值
        x_max = np.max(joints[:, :, 0])
        y_max = np.max(joints[:, :, 1])
        y_min = np.min(joints[:, :, 1])
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
         #x轴方向调整：
        ax.xaxis.set_ticks_position("top") #将x轴的位置设置在顶部
        ax.invert_xaxis() #x轴反向

        #y轴方向调整：
        ax.yaxis.set_ticks_position("right") #将y轴的位置设置在右边
        ax.invert_yaxis() #y轴反向
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_title('legrest animation', fontsize=14)
         # 设置坐标轴范围
        margin = 5
        ax.set_xlim(0-margin, x_max+margin)
        ax.set_ylim(y_min-margin, y_max+margin)

         #x轴方向调整：
        ax.xaxis.set_ticks_position("top") #将x轴的位置设置在顶部
        ax.invert_xaxis() #x轴反向

        #y轴方向调整：
        ax.yaxis.set_ticks_position("right") #将y轴的位置设置在右边
        ax.invert_yaxis() #y轴反向
        

        # ============================================================
        # 在初始化时创建标题，而不是在 animate 中
        # ============================================================
        title_text = ax.text(0.5, 0.0, f'drive_angle: 0.0 deg', 
                             transform=ax.transAxes, 
                             fontsize=12,
                             horizontalalignment='center',
                             verticalalignment='top')
            
        # 初始化绘图元素
        line_links, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
        line_driver, = ax.plot([], [], 'r-', linewidth=3, alpha=0.9)
        points_scatter, = ax.plot([], [], 'ko', markersize=8)
        end_effector, = ax.plot([], [], 'go', markersize=12, markerfacecolor='g')
        fixed_point, = ax.plot([], [], 'rs', markersize=12, markerfacecolor='r')
        path_line, = ax.plot([], [], 'g--', linewidth=1.5, alpha=0.5)

        # 2. 文本标注（预先创建7个）
        texts = []
        for idx in range(7):
            text = ax.text(0, 0, f'P{idx+1}', fontsize=9, fontweight='bold')
            texts.append(text)
            
        # 存储末端路径
        end_path = []
        
        def animate(frame):
            # 计算当前角度
            points = joints[frame]
            current_angle = driver_angles[frame]
            
            # 更新连杆
            link_x = []
            link_y = []
            for i, j in self.link_definitions:
                link_x.extend([points[i, 0], points[j, 0], np.nan])
                link_y.extend([points[i, 1], points[j, 1], np.nan])
            line_links.set_data(link_x, link_y)
            
            # 更新驱动杆 (红色高亮)
            driver_idx = self.driver_link_idx-1
            i, j = self.link_definitions[driver_idx]
            line_driver.set_data([points[i, 0], points[j, 0]], 
                                [points[i, 1], points[j, 1]])
            
            # 更新点
            points_scatter.set_data(points[:, 0], points[:, 1])
            
            # 更新固定点
            fixed_point.set_data([points[0, 0]], [points[0, 1]])
            
            # 更新末端执行器
            end_effector.set_data([points[self.end_effector_idx[0], 0]], 
                                 [points[self.end_effector_idx[0], 1]])
            
            # 更新末端路径
            end_path.append(points[self.end_effector_idx[0], :])
            if len(end_path) > 1:
                path_data = np.array(end_path)
                path_line.set_data(path_data[:, 0], path_data[:, 1])

            # # 添加标注
            # for idx, (x, y) in enumerate(points):
            #     ax.annotate(f'P{idx+1}', (x+1, y+1), fontsize=9, fontweight='bold')

            # ===== 更新文本标注 =====
            for idx, (x, y) in enumerate(points):
                texts[idx].set_position((x+1, y+1))
                
            # 更新标题（使用已有的对象，不创建新的）
            title_text.set_text(f'drive_angle: {np.degrees(current_angle):.1f} deg')
            
            return line_links, line_driver, points_scatter, end_effector, fixed_point, path_line, title_text, *texts
        
        # 创建动画
        ani = FuncAnimation(fig, animate, frames=joints.shape[0], 
                           interval=interval, blit=True)
        
        plt.tight_layout()
        plt.show()
        
        return ani
    
    def plot_configuration(self, points: np.ndarray = None, title: str = "legrest layout"):
        """
        绘制机构的当前配置
        
        参数:
        points: 点的坐标数组，如果为None则使用当前点
        title: 图形标题
        """
        if points is None:
            points = self.points_current
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        #
        #x轴方向调整：
        ax.xaxis.set_ticks_position("top") #将x轴的位置设置在顶部
        ax.invert_xaxis() #x轴反向

        #y轴方向调整：
        ax.yaxis.set_ticks_position("right") #将y轴的位置设置在右边
        ax.invert_yaxis() #y轴反向

        
        # 绘制所有连杆
        for idx, (i, j) in enumerate(self.link_definitions):
            if idx == self.driver_link_idx-1:
                # 驱动杆用红色高亮
                ax.plot([points[i, 0], points[j, 0]], 
                       [points[i, 1], points[j, 1]], 
                       'r-', linewidth=3, alpha=0.9, label='driver' if idx == 0 else "")
            else:
                ax.plot([points[i, 0], points[j, 0]], 
                       [points[i, 1], points[j, 1]], 
                       'b-', linewidth=2, alpha=0.7)
        
        # 绘制所有点
        ax.scatter(points[:, 0], points[:, 1], color='black', s=80, zorder=5)
        
        # 标注点编号
        for idx, (x, y) in enumerate(points):
            ax.annotate(f'P{idx+1}', (x+0.5, y+0.5), fontsize=8, fontweight='bold')
        
        # 特殊标记固定点和末端
        for i in self.fixed_point_idx:
            ax.scatter(points[i, 0], points[i, 1], color='red', s=150, 
                  marker='s', label="fix_pin", zorder=6)
        for i in self.end_effector_idx:
            ax.scatter(points[i, 0], points[i, 1], 
                      color='green', s=150, marker='D', label='end_effector', zorder=6)
            
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax

#新建一个类用于评估pymoo评估函数的类, 继承自LegRestMechanism
class LegRestValuation(LegRestMechanism):
    def __init__(self, points: np.ndarray):
        super().__init__(points)
        self.start_angle = np.pi/2 - self.link_angles[3]
        #执行self.forward_kinematics_interval将所有的点记录下来,用于评估
        _, self.interval_points = self.forward_kinematics_interval((self.start_angle, np.pi/2), np.deg2rad(1))

    def get_package_size(self):
        """
        根据设计变量计算机构在收回状态下的包络空间大小。
        这是第一个优化目标，需要量化 '最小化package'。
        """
        # ===== 在这里计算机构包络（Package），使用初始的状态的最大值，最小值，计算面积 = ====
        x_max = np.max(self.interval_points[0][:,0])
        x_min = np.min(self.interval_points[0][:,0])

        y_max = np.max(self.interval_points[0][:,1])
        y_min = np.min(self.interval_points[0][:,1])
        
        width = x_max - x_min  
        height = y_max - y_min
        return width * height      # 目标是最小化这个值
        
    def fit_to_lines(self):
        """
        评估末端连杆(5，7连成的杆)在几个关键工作位置时，与目标直线的匹配程度。
        这是第二个优化目标，需要量化 '与三个面最相匹配'。
        """
        # ===== 在这里计算面匹配误差 =====
        #单条线段为[[a1,a2],[b1,b2]]
        objective_lines = np.array([[[168, -4], [221.67, 68.48]], [[193.5, -40.5], [260.1, 20.3]]])  #需要拟合的位置

        #5. 7转轴点所在的位置,所有点的位置
        object_point_1 = self.interval_points[:, 4, :]
        object_point_2 = self.interval_points[:, 6, :]
               
        total_error = 0.0
        for i in range(objective_lines.shape[0]):
                        
            # 计算末端连杆上几个关键点（如两端点）在空间中的位置
            #左端点与目标点求欧氏距离
            distance_1 = np.linalg.norm(object_point_1 - objective_lines[i][0], axis = 1)
            #右端点与目标点求欧氏距离
            distance_2 = np.linalg.norm(object_point_2 - objective_lines[i][1], axis = 1)
            #距离相加并取出最小值作为与目标距离的

            total_error += np.min(distance_1 + distance_2)
        return total_error
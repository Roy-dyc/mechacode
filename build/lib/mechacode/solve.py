from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population
import numpy as np
from .mechanism import LegRestValuation


#using pymoo to problem definition and optimize 

class LegRestOptimizationProblem(ElementwiseProblem):
    """
    腿托机构多目标优化问题
    目标: 最小化包络面积 + 最小化末端位置误差
    """
    
    def __init__(self):
        # 14个设计变量: 7个点 × 2个坐标
        n_var = 14
        # 目标数量: 2个
        n_obj = 2
        # 约束数量: 0（使用惩罚方法处理）
        n_ieq_constr = 16 + 21 # 几何约束(x方向8个 + y方向8个), 最小间距约束(7个点两个组合)
        # 变量边界（根据你的设计空间设置）
        # 这里给出示例边界，需要根据实际情况调整
        xl = np.array([
            # P1 (固定点)
            0, 0,
            # P2
            0, 0,
            # P3
            0, 0,
            # P4
            0, 0,
            # P5 (末端)
            0, 0,
            # P6
            0, 0,
            # P7
            0, 0
        ])
        
        xu = np.array([
            # P1
            70, 240,
            # P2
            70, 240,
            # P3
           70, 240,
            # P4
            70, 240,
            # P5
           70, 240,
            # P6
            70, 240,
            # P7
            70, 240
        ])
        
        super().__init__(n_var=n_var,
                        n_obj=n_obj,
                        n_ieq_constr=n_ieq_constr,
                        xl=xl,
                        xu=xu)
        
        # 保存最优解历史
        self.history = []
        self.min_distance = 20
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        评估函数
        
        参数:
        x: 设计变量（一维数组，14个值）
        out: 输出字典
        """
        # 1. 将一维变量重塑为7个点
        points = x.reshape(7, 2)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
                
        # 3. 创建机构评估对象
        mechanism = LegRestValuation(points)
        
        # 4. 计算目标函数
        f1 = mechanism.get_package_size()   # 最小化包络
        f2 = mechanism.fit_to_lines()        # 最小化位置误差
        
        # 5. 输出结果
        out["F"] = np.array([f1, f2])  # 形状 (2,)

        # 记录目标值（用于调试）
        # print(f"f1 (包络面积): {f1:.2f}", end = "  ")
        # print(f"f2 (位置误差): {f2:.2f}")
            
        # 6. 记录历史（可选）
        self.history.append({
            'points': points.copy(),
            'objectives': [f1, f2]
        })

         # 加入约束
        g = []
        
        # x方向约束对
        x_pairs = [
            (1, 0), (0, 3), (3, 4), (4, 6),  # x2<x1<x4<x5<x7
            (1, 2), (2, 3), (3, 5), (5, 6)   # x2<x3<x4<x6<x7
        ]
        for i, j in x_pairs:
            g.append(x_coords[i] - x_coords[j])
        
        # y方向约束对
        y_pairs = [
            (0, 1), (1, 2),  # y1<y2; y2<y3
            (0, 3), (3, 2),  # y1<y4; y4<y3
            (4, 3), (4, 6),  # Y5<y4, Y5< y7
            (3, 5), (6, 5),  # y4<y6, y7<y6
        ]
        for i, j in y_pairs:
            g.append(y_coords[i] - y_coords[j])


        #增加两点之间min_distance的约束
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(points[i] - points[j])
                # 约束: dist >= min_distance
                # 转换为 g <= 0: min_distance - dist <= 0
                g.append(self.min_distance - dist)

        
        
        #输出约束
        out["G"] = np.array(g)
        
    def create_initial_population(self, pop_size=100):
        """
        创建满足几何约束的初始种群（纯规则生成）
        
        参数:
        pop_size: 种群大小
        
        返回:
        Population: 初始种群
        """
        all_individuals = []
        #设置两点之间的最小距离
        min_distance = 20
        
        # 分层生成 pop_size 个个体
        pop_iter_num = 0
        while pop_iter_num < pop_size:
            valid_points = self._generate_valid_points()
            if self.__check_min_distance(valid_points, min_distance):
                all_individuals.append(valid_points.flatten())
                pop_iter_num += 1
        
        # 转换为数组并打乱
        all_individuals = np.array(all_individuals)
        np.random.shuffle(all_individuals)
        
        print(f"✅ 初始种群生成完成: {len(all_individuals)} 个个体")
        print(f"   - 全部由分层规则生成")
        
        return Population.new(X=all_individuals)
    def __check_min_distance(self, points, min_distance = 15):
        #检查两点之间的距离是否大于distance
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(points[i] - points[j]) < min_distance:
                     #如不满足两点之间的距离句跳出循环
                    return False
        return True #全部满足最小距离检测
    
    def get_best_solution(self):
        """获取最优解"""
        if not self.history:
            return None
        
        # 在历史中找到目标值最小的解
        best_idx = np.argmin([h['objectives'][0] + h['objectives'][1] 
                             for h in self.history])
        return self.history[best_idx]

    def _generate_valid_points(self) -> np.ndarray:
        """
        分层生成一组满足所有约束的7个点
        
        核心思想：先确定 P4（枢纽），再向四周扩展
        """
        # 定义搜索空间
        X_MIN, X_MAX = 0, 70
        Y_MIN, Y_MAX = 0, 240
        
        # ============================================================
        # Step 1: 确定 P4（枢纽点）—— 整个机构的骨架
        # ============================================================
        # P4 应该位于空间的中间偏右区域
        x4 = np.random.uniform(10, 60)
        y4 = np.random.uniform(40, 200)
        
        # ============================================================
        # Step 2: 确定 P1（固定点，在 P4 的左下方）
        # ============================================================
        # 约束: x1 < x4, y1 < y4
        x1_min = X_MIN + 20
        x1_max = x4 - 5
        x1 = np.random.uniform(x1_min, x1_max)
        
        y1_min = Y_MIN + 5
        y1_max = y4 - 5
        y1 = np.random.uniform(y1_min, y1_max)
        
        # ============================================================
        # Step 3: 确定 P3（顶部点，在 P4 的上方）
        # ============================================================
        # 约束: y3 > y4, 且 x3 需要满足 x2 < x3 < x4
        # 这里先确定 y3 和 x3 的范围
        y3_min = y4 + 5
        y3_max = Y_MAX - 5
        y3 = np.random.uniform(y3_min, y3_max)
        
        # x3 在 x2 和 x4 之间，但 x2 还没确定，先给一个合理范围
        x3_min = X_MIN + 20
        x3_max = x4 - 5
        x3 = np.random.uniform(x3_min, x3_max)
        
        # ============================================================
        # Step 4: 确定 P2（中间层左侧）
        # ============================================================
        # 约束: x2 < x1, y1 < y2 < y3
        # 同时 x2 < x3
        x2_min = X_MIN + 5
        x2_max = min(x1 - 5, x3 - 5)
        x2 = np.random.uniform(x2_min, x2_max)
        
        y2_min = y1 + 5
        y2_max = y3 - 5
        y2 = np.random.uniform(y2_min, y2_max)
        
        # ============================================================
        # Step 6: 确定 P6（顶部右侧）
        # ============================================================
        # 约束: x4 < x6 < x7, 且 y6 > y2, y6 > y4, y6 > y7
        x6_min = x4 + 5
        x6_max = X_MAX - 20
        x6 = np.random.uniform(x6_min, x6_max)
        
        y6_min = y4 + 5
        y6_max = Y_MAX - 5
        y6 = np.random.uniform(y6_min, y6_max)
        
        # ============================================================
        # Step 7: 确定 P5（末端，最低层）
        # ============================================================
        # 约束: x3 < x4 < x5 < x7, 且 y5 < y2, y5 < y4, y5 < y7
        x5_min = x4 + 5
        x5_max = X_MAX - 20
        x5 = np.random.uniform(x5_min, x5_max)
        
        y5_min = Y_MIN + 5
        y5_max = y4 - 5
        y5 = np.random.uniform(y5_min, y5_max)

        # ============================================================
        # Step 5: 确定 P7（中间层右侧）
        # ============================================================
        # 约束: x4 < x6 < x7, 且 y1 < y7 < y3
        # 同时 y7 < y6
        x7_min = max(x4, x5) + 5
        x7_max = X_MAX - 5
        x7 = np.random.uniform(x7_min, x7_max)
        
        y7_min = y4 + 5
        y7_max = y6 - 5
        y7 = np.random.uniform(y7_min, y7_max)
        
        # ============================================================
        # Step 8: 验证所有约束
        # ============================================================
        points = np.array([
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4],
            [x5, y5],
            [x6, y6],
            [x7, y7]
        ])
        
        # 可选：验证所有约束
        # assert self._validate_points(points), "生成的点不满足约束"
        
        return points

    def _validate_points(self, points: np.ndarray) -> bool:
        """
        验证点是否满足所有几何约束
        
        参数:
        points: 7x2的坐标数组，每行代表一个点
        
        返回:
        bool: 是否满足所有约束
        """
        x = points[:, 0]
        y = points[:, 1]
        
        # ============================================================
        # X 方向约束
        # ============================================================
        # 约束1: x2 < x1 < x4 < x5 < x7
        if not (x[1] < x[0] < x[3] < x[4] < x[6]):
            return False
        
        # 约束2: x2 < x3 < x4 < x6 < x7
        if not (x[1] < x[2] < x[3] < x[5] < x[6]):
            return False
        
        # ============================================================
        # Y 方向约束
        # ============================================================
        # y1 < y2
        if not (y[0] < y[1]):
            return False
        
        # y2 < y3
        if not (y[1] < y[2]):
            return False
        
        # y1 < y4
        if not (y[0] < y[3]):
            return False
        
        # y4 < y3
        if not (y[3] < y[2]):
            return False
        
        # y5 < y4
        if not (y[4] < y[3]):
            return False
        
        # y5 < y7
        if not (y[4] < y[6]):
            return False
        
        # y4 < y6
        if not (y[3] < y[5]):
            return False
        
        # y7 < y6
        if not (y[6] < y[5]):
            return False
        
        return True

    def verify_population(self, population):
        """
        验证种群中的所有个体是否都满足约束
        
        参数:
        population: Population 对象或 numpy 数组
        
        返回:
        dict: 验证结果统计
        """
        # 如果传入的是 Population 对象
        if hasattr(population, 'get'):
            X = population.get("X")
        else:
            X = population
        
        n_individuals = X.shape[0]
        valid_count = 0
        invalid_indices = []
        
        for i in range(n_individuals):
            points = X[i].reshape(7, 2)
            if self._validate_points(points):
                valid_count += 1
            else:
                invalid_indices.append(i)
        
        result = {
            'total': n_individuals,
            'valid': valid_count,
            'invalid': n_individuals - valid_count,
            'valid_ratio': valid_count / n_individuals * 100,
            'invalid_indices': invalid_indices
        }
        
        return result


def optimize_legrest(pop_size=100, n_gen=200, seed=1):
    """
    执行腿托机构优化
    
    参数:
    pop_size: 种群大小
    n_gen: 迭代代数
    seed: 随机种子
    
    返回:
    result: 优化结果对象
    problem: 优化问题对象
    """
    # 1. 创建问题
    problem = LegRestOptimizationProblem()

    # 创建包含guess的初始种群
    initial_pop = problem.create_initial_population(pop_size=100)

    
    
    # 2. 配置NSGA-II算法
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=initial_pop,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )
    
    # 3. 执行优化
    print("=" * 50)
    print("开始腿托机构多目标优化")
    print(f"种群大小: {pop_size}")
    print(f"迭代代数: {n_gen}")
    print("=" * 50)
    
    result = minimize(problem,
                      algorithm,
                      ('n_gen', n_gen),
                      seed=seed,
                      verbose=True)
    
    print("\n" + "=" * 50)
    print("优化完成!")
    print(f"找到 {len(result.X)} 个帕累托最优解")
    print("=" * 50)
    
    return result, problem
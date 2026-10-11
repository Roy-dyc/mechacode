#此包包含一些结果的分析与可视化

import matplotlib.pyplot as plt
import numpy as np
from .mechanism import LegRestMechanism
import json

def analyze_results(result, problem):
    """分析优化结果"""
    print("\n" + "=" * 50)
    print("结果分析")
    print("=" * 50)
    
    # 提取所有解
    solutions = []
    for i in range(len(result.X)):
        points = result.X[i].reshape(7, 2)
        f1, f2 = result.F[i]
        
        solutions.append({
            'index': i,
            'points': points,
            'package': f1,
            'fit_error': f2,
            'total': f1 + f2
        })
    
    # 按总目标排序
    solutions.sort(key=lambda s: s['total'])
    
    # 显示前3个最优解
    print("\n前3个最优解（按包络+误差排序）:")
    for i, sol in enumerate(solutions[:3]):
        print(f"\n方案 {i+1}:")
        print(f"  包络面积: {sol['package']:.3f}")
        print(f"  位置误差: {sol['fit_error']:.3f}")
        print(f"  加权和: {sol['total']:.3f}")
        print("  关键点:")
        points = sol['points']
        print(f"    P1 (固定): ({points[0,0]:.2f}, {points[0,1]:.2f})")
        print(f"    P4 (驱动): ({points[3,0]:.2f}, {points[3,1]:.2f})")
        print(f"    P5 (末端): ({points[4,0]:.2f}, {points[4,1]:.2f})")
    
    # 选择最佳方案（第一个）
    best_solution = solutions[0]
    best_points = best_solution['points']
    best_objectives = [best_solution['package'], best_solution['fit_error']]
    
    return best_points, best_objectives, solutions

def plot_pareto_front(result):
    """绘制帕累托前沿"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 提取目标值
    F = result.F
    ax.scatter(F[:, 0], F[:, 1], color='blue', alpha=0.6, s=50)
    
    # 标记最佳点（加权和最小）
    total = F[:, 0] + F[:, 1]
    best_idx = np.argmin(total)
    ax.scatter(F[best_idx, 0], F[best_idx, 1], 
              color='red', s=200, marker='*', 
              label='Best solution ')
    
    ax.set_xlabel('package (minimize)', fontsize=12)
    ax.set_ylabel('Position error (minimize)', fontsize=12)
    ax.set_title('Pareto Front', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_optimized_mechanism(points, title="optimized_legrest_mechanism"):
    """绘制优化后的机构"""
    #from legrest_mechanism import LegRestMechanism  # 你的机构类
    
    mechanism = LegRestMechanism(points)
    mechanism.plot_configuration(title=title)

def save_to_json(solutions, filename = r".\solutions.json"):
    """
    将包含 NumPy 数组的解决方案列表转换为可序列化的格式
    """
    serializable_solutions = []
    
    for sol in solutions:
        # 复制字典，避免修改原数据
        new_sol = sol.copy()
        
        # ✅ 将 NumPy 数组转换为列表
        if 'points' in new_sol and isinstance(new_sol['points'], np.ndarray):
            new_sol['points'] = new_sol['points'].tolist()
        
        serializable_solutions.append(new_sol)
    # 保存为 JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_solutions, f, indent=4, ensure_ascii=False)
        

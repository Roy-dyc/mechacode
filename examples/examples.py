#写一个例子
from mechacode.solve import LegRestOptimizationProblem
from mechacode.solve import optimize_legrest
from mechacode.results import analyze_results
from mechacode.results import plot_pareto_front




def main():
    # 创建问题实例
    problem = LegRestOptimizationProblem()

    # 生成初始种群
    initial_pop = problem.create_initial_population(pop_size=100)

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
    print("="*50)
    print("使用NSGAII进行优化求解")
    print("="*50)
    #使用NSGA，在初始种群的情况下进行优化求解
    result= optimize_legrest(pop_size=100, n_gen=200, seed=1)
    print("="*50)
    print("开始结构分析")
    print("="*50)
    #分析结果
    best_points, best_objectives, solutions = analyze_results(result, problem)  
    plot_pareto_front(result)

    


if __name__ == "__main__":
    main()
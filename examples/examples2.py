from mechacode.solve import LegRestOptimizationProblem
from mechacode.mechanism import LegRestMechanism
import numpy as np


def main():
    # 创建问题实例
    problem = LegRestOptimizationProblem()

    # 生成初始种群
    initial_pop = problem.create_initial_population(pop_size=100)

    # 验证所有个体
    pop_result = problem.verify_population(initial_pop)

    array_X = initial_pop.get("X")
    single_choice = array_X[0].reshape(7,2)
    

    

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

    
    print("=" * 50)
    print("如下是演示部分：")
    print("=" * 50)
    print(single_choice)
    my_legrest = LegRestMechanism(single_choice)
    my_legrest.plot_configuration()
    my_legrest.animate_mechanism([np.pi/2 - my_legrest.link_angles[3], np.pi/2])


if __name__ == "__main__":
    main()
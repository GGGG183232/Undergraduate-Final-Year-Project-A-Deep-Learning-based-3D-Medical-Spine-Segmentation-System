def min_edit_time(n, k, tasks):
    min_time = float('inf')
    from itertools import combinations
    # 生成所有可能的 k 个任务的组合，保持顺序
    indices = list(range(n))
    combos = combinations(indices, k)
    for combo in combos:
        selected_tasks = [tasks[i] for i in combo]
        # 计算前缀和
        prefix_sum = [0] * (k + 1)
        for i in range(k):
            prefix_sum[i + 1] = prefix_sum[i] + selected_tasks[i]
        # 遍历所有可能的分割点
        for i in range(k + 1):
            time1 = prefix_sum[i]
            time2 = prefix_sum[k] - time1
            min_time = min(min_time, max(time1, time2))
    return min_time


# 读取输入
T = int(input())
results = []
for _ in range(T):
    n, k = map(int, input().split())
    tasks = list(map(int, input().split()))
    results.append(str(min_edit_time(n, k, tasks)))

# 输出结果
print("\n".join(results))

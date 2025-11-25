import random

def generate_random_numbers(target: float, lower_delta: float, upper_delta: float, count: int):
    """
    生成围绕 target 的随机数，可手动指定下界浮动和上界浮动。

    :param target: 目标数字
    :param lower_delta: 向下浮动多少（负方向），例如 0.2 -> 下界 = target - 0.2
    :param upper_delta: 向上浮动多少（正方向），例如 0.5 -> 上界 = target + 0.5
    :param count: 要生成的数字个数
    :return: 随机数列表
    """
    low = target - lower_delta
    high = target + upper_delta
    return [random.uniform(low, high) for _ in range(count)]

if __name__ == "__main__":
    # 示例：围绕目标值 10，向下浮动 1.2，向上浮动 0.3，生成 5 个随机数
    nums = generate_random_numbers(target=66.54, lower_delta=3, upper_delta=1, count=6)

    for i in nums:
        print(i)

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 参数定义
L = 100         # 道路长度（cells）
N = 20          # 车辆数
V_max = 5       # 最大速度
p = 0.3         # 随机减速概率
T = 100         # 仿真时间步数

# 车辆初始化（随机分布）
positions = np.sort(np.random.choice(L, N, replace=False))  # 随机分布
velocities = np.random.randint(0, V_max + 1, N)  # 随机初始速度

# 存储演变数据
road_history = np.zeros((T, L), dtype=int)

# 定义 NaSch 模型的更新步骤
def update_positions(positions, velocities, L):
    return (positions + velocities) % L  # 确保环形

def calculate_distances(positions, L):
    distances = np.roll(positions, -1) - positions - 1
    distances[-1] += L  # 处理环形边界条件
    return distances

def accelerate(velocities, V_max):
    return np.minimum(velocities + 1, V_max)

def brake(velocities, distances):
    return np.minimum(velocities, distances)

def random_slow_down(velocities, p):
    slow_down = np.random.rand(len(velocities)) < p
    velocities[slow_down] = np.maximum(velocities[slow_down] - 1, 0)
    return velocities

# 运行 NaSch 模型
for t in range(T):
    road = np.zeros(L)
    
    # 计算车距
    distances = calculate_distances(positions, L)
    
    # 1. 加速
    velocities = accelerate(velocities, V_max)
    
    # 2. 刹车（避免碰撞）
    velocities = brake(velocities, distances)
    
    # 3. 随机慢速（概率 p）
    velocities = random_slow_down(velocities, p)
    
    # 4. 位置更新
    positions = update_positions(positions, velocities, L)
    
    # 记录车流
    road[positions] = 1
    road_history[t] = road

# 结果可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(road_history, cmap="Greys", aspect="auto")
ax.set_xlabel("位置")
ax.set_ylabel("时间步")
ax.set_title("Nagel-Schreckenberg 交通流模拟")
plt.show()

# 动态可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, L)
ax.set_ylim(0, 1)
ax.set_xlabel("位置")
ax.set_title("Nagel-Schreckenberg 交通流模拟")

line, = ax.plot(positions, np.zeros(N), 'bo')

def animate(t):
    road = road_history[t]
    positions = np.where(road == 1)[0]
    line.set_xdata(positions)
    return line,

ani = animation.FuncAnimation(fig, animate, frames=T, interval=100, blit=True)
plt.show()


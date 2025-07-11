import heapq
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始地图（包含捷径 0.3，但我们禁用它）
raw_grid = np.array([
    [1, 1, 1, 5, 1, 1, 0, 1],
    [1, 0, 1, 5, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 5, 1],
    [1, 5, 5, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1],
    [5, 0, 0, 5, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 5, 1, 1, 1, 1]
])
start, goal = (0, 0), (7, 7)

# 禁用捷径（没有设置为 0.3，因此这里不修改）
processed_grid = np.copy(raw_grid)

# 可视化映射：0=障碍，1=普通，2=拥堵
visual_grid = np.zeros_like(processed_grid, dtype=int)
visual_grid[processed_grid == 1] = 1      # 普通道路
visual_grid[processed_grid == 5] = 2      # 拥堵道路
visual_grid[processed_grid == 0] = 0      # 障碍

# 定义颜色映射
cmap = ListedColormap(['red', 'lightblue', 'gold'])  # 0=障碍, 1=普通, 2=拥堵

# 绘制地图
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.imshow(visual_grid, cmap=cmap, vmin=0, vmax=2)

# 添加网格线
ax.set_xticks(np.arange(-0.5, visual_grid.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, visual_grid.shape[0], 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', size=0)

# 起点终点标记
ax.scatter(start[1], start[0], color='green', s=300, marker='o', label='仓库')
ax.scatter(goal[1], goal[0], color='red', s=300, marker='*', label='客户家')

# 图例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='小车', markerfacecolor='lime', markersize=10),
    plt.Line2D([0], [0], color='red', lw=2, label='最优路径'),
    plt.Rectangle((0, 0), 1, 1, fc='lightblue', edgecolor='gray', label='普通道路'),
    plt.Rectangle((0, 0), 1, 1, fc='gold', edgecolor='gray', label='拥堵路段'),
    plt.Rectangle((0, 0), 1, 1, fc='red', edgecolor='gray', label='障碍物'),
]
ax.legend(handles=legend_elements, bbox_to_anchor=(1.35, 1), fontsize=10)

# 启发函数
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* 算法
def a_star(grid, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_set = [(fscore[start], start)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor] == 0:
                    continue
                tentative_g = gscore[current] + grid[neighbor]
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, float('inf')):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in open_set]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (fscore[neighbor], neighbor))
    return []

# 路径规划
path = a_star(processed_grid, start, goal)
print("最优路径坐标:", path)

# 动画组件
car = ax.scatter([], [], color='lime', s=400, zorder=10)
line, = ax.plot([], [], 'r-', linewidth=2)

def init():
    car.set_offsets([[start[1], start[0]]])
    line.set_data([], [])
    return car, line

def update(frame):
    x, y = path[frame][1], path[frame][0]
    car.set_offsets([[x, y]])
    line.set_data([p[1] for p in path[:frame+1]], [p[0] for p in path[:frame+1]])
    return car, line

# 生成动画
ani = FuncAnimation(plt.gcf(), update, frames=len(path), init_func=init, blit=True, interval=500)

plt.title("A*路径规划", fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# 保存为GIF文件（确保安装 pillow）
ani.save('a_star_path.gif', writer='pillow', fps=2)

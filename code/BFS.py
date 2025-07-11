import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import deque
import os

def bfs_path(graph, start, end):
    queue = deque([start])  # BFS 队列
    visited = {start}       # 记录访问过的节点
    parent = {}             # 记录前驱节点（用于回溯路径）

    while queue:
        node = queue.popleft()
        
        if node == end:  # 找到目标节点，回溯路径
            path = []
            while node is not None:
                path.append(node)
                node = parent.get(node)  # 追溯前驱节点
            return path[::-1]  # 反转得到正序路径
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = node  # 记录前驱节点

    return None  # 没有找到路径

# 示例图（邻接表表示）
graph = {
    'Arad': ['Zerind', 'Timisoara', 'Sibiu'],
    'Zerind': ['Oradea', 'Arad'],
    'Oradea': ['Sibiu', 'Zerind'],
    'Sibiu': ['Rimnicu Vilcea', 'Fagaras', 'Arad', 'Oradea'],
    'Rimnicu Vilcea': ['Sibiu', 'Pitesti', 'Craiova'],
    'Pitesti': ['Rimnicu Vilcea', 'Bucharest', 'Craiova'],
    'Bucharest': ['Urziceni', 'Giurgiu', 'Pitesti', 'Fagaras'],
    'Urziceni': ['Bucharest', 'Hirsova', 'Vaslui'],
    'Hirsova': ['Eforie', 'Urziceni'],
    'Vaslui': ['Lasi', 'Urziceni'],
    'Lasi': ['Neamt', 'Vaslui'],
    'Craiova': ['Dobreta', 'Pitesti', 'Rimnicu Vilcea'],
    'Dobreta': ['Mehadia', 'Craiova'],
    'Mehadia': ['Lugoj', 'Dobreta'],
    'Lugoj': ['Mehadia', 'Timisoara'],
    'Timisoara': ['Lugoj', 'Arad'],
    'Eforie': ['Hirsova'],
    'Giurgiu': ['Bucharest'],
    'Fagaras': ['Sibiu', 'Bucharest'],
    'Neamt': ['Lasi']
}

# 运行 BFS 计算路径
start, end = 'Arad', 'Bucharest'
path = bfs_path(graph, start, end)
print("最短路径:", path)

def visualize_bfs_path(graph, start, end):
    # 获取当前文件所在的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建images文件夹
    output_dir = os.path.join(current_dir, 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.Graph()
    for node in graph:
        for neighbor in graph[node]:
            G.add_edge(node, neighbor)

    pos = nx.spring_layout(G)  # 生成图布局
    step = 0  # 用于图片命名

    visited = set()
    queue = deque([start])
    visited.add(start)
    parent = {start: None}
    current_path = []

    while queue:
        node = queue.popleft()
        current_path.append(node)
        
        # 为每一步创建新的图
        plt.figure(figsize=(6, 4))
        # 绘制基本图形
        nx.draw(G, pos, with_labels=True, node_color="#ADD8E6", edge_color="gray", 
                node_size=700, font_size=8)
        # 绘制当前访问的节点
        nx.draw_networkx_nodes(G, pos, nodelist=current_path, 
                             node_color="orange", node_size=300)
        
        # 保存当前步骤的图片
        plt.savefig(os.path.join(output_dir, f'bfs_step_{step}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        step += 1

        if node == end:
            break  # 找到终点，结束 BFS
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = node

    # 回溯得到最终路径
    final_path = []
    node = end
    while node is not None:
        final_path.append(node)
        node = parent.get(node)
    final_path.reverse()

    # 绘制并保存最终路径图
    plt.figure(figsize=(6, 4))
    # 绘制基本图形
    nx.draw(G, pos, with_labels=True, node_color="#ADD8E6", edge_color="gray", 
            node_size=700, font_size=8, font_weight='bold')
    # 绘制最终路径节点
    nx.draw_networkx_nodes(G, pos, nodelist=final_path, 
                          node_color="red", node_size=300)
    # 绘制最终路径边
    nx.draw_networkx_edges(G, pos, 
                          edgelist=[(final_path[i], final_path[i+1]) 
                                   for i in range(len(final_path)-1)], 
                          edge_color="orange", width=2)
    plt.savefig(os.path.join(output_dir, 'bfs_final_path.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"所有图片已保存到: {output_dir}")

# 运行可视化
visualize_bfs_path(graph, 'Arad', 'Bucharest')

import heapq
import networkx as nx
import matplotlib.pyplot as plt
import os
import math

def heuristic(node, end, pos):
    """计算启发式函数（欧几里得距离）"""
    return math.sqrt((pos[node][0]-pos[end][0])**2 + (pos[node][1]-pos[end][1])**2)

def astar_shortest_path(graph, start, end, pos):
    """A*算法寻找最短路径"""
    # 优先队列：(f_score, g_score, node, path)
    open_set = [(0 + heuristic(start, end, pos), 0, start, [start])]
    heapq.heapify(open_set)
    
    # 记录已访问节点及其g_score
    visited = {start: 0}
    
    while open_set:
        _, g, node, path = heapq.heappop(open_set)
        
        if node == end:
            return path, g
        
        for neighbor, weight in graph.get(node, []):
            new_g = g + weight
            
            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                f_score = new_g + heuristic(neighbor, end, pos)
                heapq.heappush(open_set, (f_score, new_g, neighbor, path + [neighbor]))
    
    return None, float('inf')

def visualize_astar(graph, start, end):
    """A*算法可视化"""
    # 创建输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'astar_images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建图对象
    G = nx.Graph()
    for node in graph:
        for neighbor, weight in graph[node]:
            G.add_edge(node, neighbor, weight=weight)
    
    # 固定布局（确保每次运行结果一致）
    pos = nx.spring_layout(G, seed=42)
    
    # 计算启发式距离（用于显示）
    h_values = {node: heuristic(node, end, pos) for node in G.nodes()}
    
    # A*算法初始化
    open_set = [(0 + h_values[start], 0, start, [start])]
    heapq.heapify(open_set)
    visited = {start: 0}
    step = 0
    
    while open_set:
        _, g, node, path = heapq.heappop(open_set)
        
        # 绘制当前状态
        plt.figure(figsize=(10, 8))
        
        # 绘制基本图形
        nx.draw(G, pos, with_labels=True, 
               node_color="#ADD8E6",  # 浅蓝色
               edge_color="gray", 
               node_size=800,
               font_size=8)
        
        # 显示边权值
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, 
                                   edge_labels=edge_labels,
                                   font_size=7)
        
        # 显示启发式值
        node_labels = {n: f"h={h_values[n]:.1f}" for n in G.nodes()}
        pos_labels = {k: (v[0], v[1]+0.05) for k, v in pos.items()}
        nx.draw_networkx_labels(G, pos_labels, 
                               labels=node_labels,
                               font_size=7,
                               font_color='red')
        
        # 高亮已访问节点（橙色）
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=visited.keys(),
                              node_color="orange",
                              node_size=600)
        
        # 高亮当前路径（红色）
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=path,
                              node_color="red",
                              node_size=800)
        nx.draw_networkx_edges(G, pos, 
                              edgelist=[(path[i], path[i+1]) for i in range(len(path)-1)],
                              edge_color="red",
                              width=2)
        
        # 添加标题信息
        plt.title(f"A* Step {step}\nCurrent: {node} (g={g:.1f}, h={h_values[node]:.1f}, f={g+h_values[node]:.1f})\n"
                 f"Path: {' → '.join(path)}", 
                 pad=20, fontsize=10)
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, f'astar_step_{step}.png'), 
                    bbox_inches='tight', 
                    dpi=150)
        plt.close()
        step += 1
        
        if node == end:
            break
        
        for neighbor, weight in graph.get(node, []):
            new_g = g + weight
            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                f_score = new_g + h_values[neighbor]
                heapq.heappush(open_set, (f_score, new_g, neighbor, path + [neighbor]))
    
    # 绘制最终路径
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True,
           node_color="#ADD8E6",
           edge_color="gray",
           node_size=800,
           font_size=8)
    
    # 显示边权值和启发式值
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    nx.draw_networkx_labels(G, pos_labels, labels=node_labels, font_size=7, font_color='red')
    
    # 高亮最终路径（深红色）
    nx.draw_networkx_nodes(G, pos,
                          nodelist=path,
                          node_color="#FF4500",  # 橙红色
                          node_size=800)
    nx.draw_networkx_edges(G, pos,
                          edgelist=[(path[i], path[i+1]) for i in range(len(path)-1)],
                          edge_color="#FF4500",
                          width=3)
    
    # 计算总代价
    total_cost = sum(graph[path[i]][0][1] for i in range(len(path)-1))
    
    plt.title(f"A* Final Path (Cost: {total_cost})\n{' → '.join(path)}", 
             pad=20, fontsize=12)
    plt.savefig(os.path.join(output_dir, 'astar_final.png'), 
                bbox_inches='tight', 
                dpi=150)
    plt.close()
    
    print(f"可视化结果已保存到: {output_dir}")
    return path, total_cost

# 使用相同的图数据
graph = {
    'Arad': [('Zerind',75), ('Timisoara',118), ('Sibiu',140)],
    'Zerind': [('Oradea',71), ('Arad',75)],
    'Oradea': [('Sibiu',151), ('Zerind',71)],
    'Sibiu': [('Rimnicu Vilcea',80), ('Fagaras',99), ('Arad',140), ('Oradea',151)],
    'Rimnicu Vilcea': [('Sibiu',80), ('Pitesti',97), ('Craiova',146)],
    'Pitesti': [('Rimnicu Vilcea',97), ('Bucharest',101), ('Craiova',138)],
    'Bucharest': [('Urziceni',85), ('Giurgiu',90), ('Pitesti',101), ('Fagaras',211)],
    'Urziceni': [('Bucharest',85), ('Hirsova',98), ('Vaslui',142)],
    'Hirsova': [('Eforie',86), ('Urziceni',98)],
    'Vaslui': [('Lasi',92), ('Urziceni',142)],
    'Lasi': [('Neamt',87), ('Vaslui',92)],
    'Craiova': [('Dobreta',120), ('Pitesti',138), ('Rimnicu Vilcea',146)],
    'Dobreta': [('Mehadia',75), ('Craiova',120)],
    'Mehadia': [('Lugoj',70), ('Dobreta',75)],
    'Lugoj': [('Mehadia',70), ('Timisoara',111)],
    'Timisoara': [('Lugoj',111), ('Arad',118)],
    'Eforie': [('Hirsova',86)],
    'Giurgiu': [('Bucharest',90)],
    'Fagaras': [('Sibiu',99), ('Bucharest',211)],
    'Neamt': [('Lasi',87)]
}

# 运行A*算法和可视化
start, end = 'Arad', 'Bucharest'
path, cost = visualize_astar(graph, start, end)
print(f"A*找到的最短路径: {path}")
print(f"路径总代价: {cost}") 
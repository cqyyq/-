import heapq 
import networkx as nx
import matplotlib.pyplot as plt
import os
import math

def heuristic(node, end, pos):
    """计算启发式函数（欧几里得距离）"""
    return math.sqrt((pos[node][0]-pos[end][0])**2 + (pos[node][1]-pos[end][1])**2)

def greedy_search(graph, start, end, pos):
    """贪心算法寻找路径"""
    # 优先队列：按启发式值排序
    open_set = [(heuristic(start, end, pos), start, [start])]
    heapq.heapify(open_set)
    visited = set()
    
    while open_set:
        _, node, path = heapq.heappop(open_set)
        
        if node == end:
            # 计算实际路径代价
            cost = sum(graph[path[i]][0][1] for i in range(len(path)-1))
            return path, cost
        
        if node in visited:
            continue
        visited.add(node)
        
        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(open_set, (heuristic(neighbor, end, pos), neighbor, path + [neighbor]))
    
    return None, float('inf')

def visualize_greedy(graph, start, end):
    """贪心算法可视化"""
    # 创建输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'greedy_images')
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
    
    # 算法初始化
    open_set = [(h_values[start], start, [start])]
    heapq.heapify(open_set)
    visited = set()
    step = 0
    
    while open_set:
        h, node, path = heapq.heappop(open_set)
        
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
                              nodelist=list(visited),
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
        plt.title(f"Greedy Search Step {step}\nCurrent: {node} (h={h:.1f})\n"
                 f"Path: {' → '.join(path)}", 
                 pad=20, fontsize=10)
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, f'greedy_step_{step}.png'), 
                    bbox_inches='tight', 
                    dpi=150)
        plt.close()
        step += 1
        
        if node == end:
            break
        
        if node in visited:
            continue
        visited.add(node)
        
        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(open_set, (h_values[neighbor], neighbor, path + [neighbor]))
    
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
    
    plt.title(f"Greedy Final Path (Cost: {total_cost})\n{' → '.join(path)}", 
             pad=20, fontsize=12)
    plt.savefig(os.path.join(output_dir, 'greedy_final.png'), 
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

# 运行贪心算法和可视化
start, end = 'Arad', 'Bucharest'
path, cost = visualize_greedy(graph, start, end)
print(f"贪心算法找到的路径: {path}")
print(f"路径总代价: {cost}")
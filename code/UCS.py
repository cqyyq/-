import heapq
import networkx as nx
import matplotlib.pyplot as plt
import time
import os

def ucs_shortest_path(graph, start, end):
    # 优先队列：存储 (当前路径代价, 当前节点, 当前路径)
    pq = [(0, start, [start])]  
    visited = set()  # 记录访问过的节点

    while pq:
        cost, node, path = heapq.heappop(pq)  # 取出当前代价最小的节点

        if node in visited:
            continue  # 避免重复访问
        visited.add(node)

        if node == end:  # 找到目标节点
            return path, cost  

        for neighbor, weight in graph.get(node, []):  
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))  

    return None, float('inf')  # 无法到达终点

# 示例：带权图
graph = {
        'Arad': [('Zerind',75), ('Timisoara',118), ('Sibiu',140)],
    'Zerind': [('Oradea',71), ('Arad',75)],
    'Oradea': [('Sibiu',151), ('Zerind',71)],
    'Sibiu': [('Rimnicu Vilcea',80), ('Fagaras',90), ('Arad',140), ('Oradea',151)],
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

# 计算最短路径
start, end = 'Arad', 'Bucharest'
path, cost = ucs_shortest_path(graph, start, end)
print("最短路径:", path)
print("最短路径代价:", cost)

def visualize_ucs(graph, start, end):
    # 获取当前文件所在的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建images文件夹
    output_dir = os.path.join(current_dir, 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.Graph()
    for node in graph:
        for neighbor, weight in graph[node]:
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)  
    step = 0  # 用于图片命名

    # UCS 搜索过程
    pq = [(0, start, [start])]
    visited = set()
    path = []

    while pq:
        cost, node, current_path = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)
        path = current_path

        # 为每一步创建新的图
        plt.figure(figsize=(6, 4))
        # 绘制基本图形
        nx.draw(G, pos, with_labels=True, node_color="#ADD8E6", edge_color="gray", node_size=700, font_size=8)
        edge_labels = {(u, v): w for u, v, w in G.edges(data="weight")}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

        # 绘制当前访问的节点
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="orange", node_size=300)
        
        # 保存当前步骤的图片
        plt.savefig(os.path.join(output_dir, f'ucs_step_{step}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        step += 1

        if node == end:
            break  # 终点找到

        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))

    # 绘制并保存最终路径图
    plt.figure(figsize=(6, 4))
    # 绘制基本图形
    nx.draw(G, pos, with_labels=True, node_color="#ADD8E6", edge_color="gray", node_size=700, font_size=8)
    edge_labels = {(u, v): w for u, v, w in G.edges(data="weight")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    # 绘制最终路径节点
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="red", node_size=300)
    # 绘制最终路径边
    nx.draw_networkx_edges(G, pos, 
                          edgelist=[(path[i], path[i+1]) for i in range(len(path)-1)], 
                          edge_color="red", width=2)
    plt.savefig(os.path.join(output_dir, 'ucs_final_path.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"所有图片已保存到: {output_dir}")

# 运行可视化
visualize_ucs(graph, 'Arad', 'Bucharest')

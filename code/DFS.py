import networkx as nx
import matplotlib.pyplot as plt
import time
import os

def dfs_path(graph, start, end, path=None):
    if path is None:
        path = []
    
    path = path + [start]  # 记录当前路径

    if start == end:  # 找到目标节点
        return path

    for neighbor in graph.get(start, []):
        if neighbor not in path:  # 避免重复访问
            new_path = dfs_path(graph, neighbor, end, path)
            if new_path:  # 如果找到路径，返回
                return new_path

    return None  # 无路径

# 示例图（无权图）
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

# 运行 DFS 计算路径
start, end = 'Arad', 'Bucharest'
dfs_result = dfs_path(graph, start, end)
print("DFS 找到的路径:", dfs_result)

def visualize_dfs(graph, start, end):
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
    path = []  # 记录当前路径
    found = False

    def dfs(node):
        nonlocal found, step
        if found:
            return
        
        path.append(node)

        # 为每一步创建新的图
        plt.figure(figsize=(6, 4))
        # 绘制基本图形
        nx.draw(G, pos, with_labels=True, node_color="#ADD8E6", edge_color="gray", 
                node_size=700, font_size=8)
        # 绘制当前访问的节点
        nx.draw_networkx_nodes(G, pos, nodelist=path, 
                             node_color="orange", node_size=300)
        
        # 保存当前步骤的图片
        plt.savefig(os.path.join(output_dir, f'dfs_step_{step}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        step += 1

        if node == end:  # 找到终点，结束搜索
            found = True
            return

        for neighbor in graph.get(node, []):
            if neighbor not in path:
                dfs(neighbor)
                if found:
                    return

        if not found:
            path.pop()  # 回溯

    dfs(start)

    # 绘制并保存最终路径图
    plt.figure(figsize=(6, 4))
    # 绘制基本图形
    nx.draw(G, pos, with_labels=True, node_color="#ADD8E6", edge_color="gray", 
            node_size=700, font_size=8, font_weight='bold')
    # 绘制最终路径节点
    nx.draw_networkx_nodes(G, pos, nodelist=path, 
                          node_color="red", node_size=300)
    # 绘制最终路径边
    nx.draw_networkx_edges(G, pos, 
                          edgelist=[(path[i], path[i+1]) 
                                   for i in range(len(path)-1)], 
                          edge_color="red", width=2)
    plt.savefig(os.path.join(output_dir, 'dfs_final_path.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"所有图片已保存到: {output_dir}")

# 运行可视化
visualize_dfs(graph, 'Arad', 'Bucharest')

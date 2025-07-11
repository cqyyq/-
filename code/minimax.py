import networkx as nx  # 用于构建与可视化有向图
import matplotlib.pyplot as plt  # 用于绘图
import os  # 添加os模块用于处理路径
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# —— 1. 定义树结构与节点属性 ——
# 树的拓扑结构
tree = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': ['H', 'I'],
    'E': ['J', 'K'],
    'F': ['L', 'M'],
    'G': ['N', 'O']
}

# 叶节点的静态评估值
leaf_values = {
    'H': -1, 'I': 4,
    'J': 2,  'K': 6,
    'L': -3, 'M': -5,
    'N': 0,  'O': 7
}

# 节点角色：True 表示 Max 层，False 表示 Min 层，None 表示叶节点
node_types = {
    'A': True,  'B': False, 'C': False,
    'D': True,  'E': True,  'F': True,  'G': True,
    'H': None,  'I': None,  'J': None,  'K': None,
    'L': None,  'M': None,  'N': None,  'O': None
}

# 存放每个节点的 Minimax 计算结果
node_values = {}

# —— 2. Minimax 算法 实现 ——
def minimax(node):
    """
    对给定节点执行 Minimax 递归计算，返回该节点的最优值。
    """
    # 如果是叶节点，直接返回静态评估
    if node_types[node] is None:
        node_values[node] = leaf_values[node]
        return leaf_values[node]
    # 否则，递归计算所有子节点
    children = tree[node]
    values = [minimax(child) for child in children]
    # Max 层取最大值，Min 层取最小值
    if node_types[node]:
        best = max(values)
    else:
        best = min(values)
    node_values[node] = best
    return best

# 执行计算
root_value = minimax('A')
print(f"根节点 A 的评估值: {root_value}")  # 期望输出 4

# —— 3. 可视化 ——  
def get_tree_layout(tree):
    """计算树的均匀布局"""
    # 首先获取每层的节点
    layers = []
    current_layer = ['A']
    while current_layer:
        layers.append(current_layer)
        next_layer = []
        for node in current_layer:
            if node in tree:
                next_layer.extend(tree[node])
        current_layer = next_layer

    # 计算每个节点的位置
    pos = {}
    for layer_idx, layer in enumerate(layers):
        y = 1 - layer_idx * (1.0 / (len(layers) - 1))  # y坐标均匀分布
        
        # 计算这一层的x坐标
        if len(layer) == 1:
            pos[layer[0]] = [0.5, y]
        else:
            for i, node in enumerate(layer):
                x = (i + 1) / (len(layer) + 1)  # x坐标均匀分布
                pos[node] = [x, y]
    
    return pos

def visualize_minimax(tree, values, types):
    """
    将 Minimax 计算结果可视化为一个分层有向图，
    节点标签包含名称与评估值，节点颜色区分 Max/Min/叶子。
    """
    G = nx.DiGraph()
    # 添加节点与边
    for parent, children in tree.items():
        G.add_node(parent)
        for child in children:
            G.add_node(child)
            G.add_edge(parent, child)

    # 节点标签与样式
    labels = {n: f"{n}\n{values[n]}" for n in G.nodes()}
    node_color = []
    for n in G.nodes():
        if types[n] is True:
            node_color.append('#a6cee3')   # Max 层：浅蓝
        elif types[n] is False:
            node_color.append('#b2df8a')   # Min 层：浅绿
        else:
            node_color.append('#fb9a99')   # 叶节点：浅红

    # 使用自定义布局
    pos = get_tree_layout(tree)
    
    plt.figure(figsize=(12, 8))
    # 首先绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    # 然后绘制节点和标签
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=2000)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', font_family='SimHei')
    
    plt.title("Minimax 博弈树可视化", fontsize=16, fontfamily='SimHei', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图片到代码所在目录
    output_path = os.path.join(SCRIPT_DIR, 'minimax_tree.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    return output_path

if __name__ == '__main__':

    # 生成可视化并保存
    output_path = visualize_minimax(tree, node_values, node_types)
    print(f"博弈树已保存为: {output_path}")

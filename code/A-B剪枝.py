import networkx as nx
import matplotlib.pyplot as plt
import os  # 添加os模块用于处理路径
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义博弈树结构
tree = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': ['H', 'I'],
    'E': ['J', 'K'],
    'F': ['L', 'M'],
    'G': ['N', 'O']
}

# 叶节点的评估值
leaf_values = {
    'H': -1, 'I': 4,
    'J': 2,  'K': 6,
    'L': -3, 'M': -5,
    'N': 0,  'O': 7
}

# 节点类型：True表示MAX层，False表示MIN层，叶节点为None
node_types = {
    'A': True,  # MAX层
    'B': False, 'C': False,  # MIN层
    'D': True,  'E': True, 'F': True, 'G': True,  # MAX层
    'H': None, 'I': None, 'J': None, 'K': None,  # 叶节点
    'L': None, 'M': None, 'N': None, 'O': None   # 叶节点
}

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

def get_all_children(node, tree):
    """获取节点的所有子节点（包括子节点的子节点）"""
    children = set()
    if node in tree:
        for child in tree[node]:
            children.add(child)
            children.update(get_all_children(child, tree))
    return children

def minimax_ab(node, is_maximizing, alpha, beta, tree_data=None, parent=None):
    """
    带Alpha-Beta剪枝的Minimax算法
    node: 当前节点
    is_maximizing: 是否为MAX层
    alpha: alpha值
    beta: beta值
    tree_data: 用于可视化的树信息
    parent: 父节点
    """
    if tree_data is None:
        tree_data = {}
    
    # 记录节点信息
    if node not in tree_data:
        tree_data[node] = {
            'value': None,
            'children': [],
            'pruned': False,
            'alpha': alpha,
            'beta': beta,
            'type': 'MAX' if is_maximizing else 'MIN'
        }
    if parent:
        tree_data[parent]['children'].append(node)

    # 如果是叶节点，返回评估值
    if node_types[node] is None:
        value = leaf_values[node]
        tree_data[node]['value'] = value
        return value

    # 如果是MAX层
    if is_maximizing:
        value = float('-inf')
        for child in tree[node]:
            child_value = minimax_ab(child, False, alpha, beta, tree_data, node)
            value = max(value, child_value)
            alpha = max(alpha, value)
            if alpha >= beta:  # 剪枝条件
                tree_data[node]['pruned'] = True
                # 标记未访问的子节点为被剪枝
                unvisited_children = set(tree[node]) - set(tree_data[node]['children'])
                for unvisited in unvisited_children:
                    tree_data[unvisited] = {
                        'value': '?',
                        'children': [],
                        'pruned': True,
                        'alpha': alpha,
                        'beta': beta,
                        'type': 'MIN'
                    }
                    # 标记未访问节点的所有子节点
                    for child in get_all_children(unvisited, tree):
                        tree_data[child] = {
                            'value': '?',
                            'children': [],
                            'pruned': True,
                            'alpha': alpha,
                            'beta': beta,
                            'type': 'MAX' if node_types.get(child, True) else 'MIN'
                        }
                break
        tree_data[node]['value'] = value
        tree_data[node]['alpha'] = alpha
        tree_data[node]['beta'] = beta
        return value
    # 如果是MIN层
    else:
        value = float('inf')
        for child in tree[node]:
            child_value = minimax_ab(child, True, alpha, beta, tree_data, node)
            value = min(value, child_value)
            beta = min(beta, value)
            if beta <= alpha:  # 剪枝条件
                tree_data[node]['pruned'] = True
                # 标记未访问的子节点为被剪枝
                unvisited_children = set(tree[node]) - set(tree_data[node]['children'])
                for unvisited in unvisited_children:
                    tree_data[unvisited] = {
                        'value': '?',
                        'children': [],
                        'pruned': True,
                        'alpha': alpha,
                        'beta': beta,
                        'type': 'MAX'
                    }
                    # 标记未访问节点的所有子节点
                    for child in get_all_children(unvisited, tree):
                        tree_data[child] = {
                            'value': '?',
                            'children': [],
                            'pruned': True,
                            'alpha': alpha,
                            'beta': beta,
                            'type': 'MIN' if node_types.get(child, False) else 'MAX'
                        }
                break
        tree_data[node]['value'] = value
        tree_data[node]['alpha'] = alpha
        tree_data[node]['beta'] = beta
        return value

def visualize_tree(tree_data):
    """将博弈树可视化为matplotlib图形"""
    G = nx.DiGraph()
    
    # 添加节点和边
    for node, data in tree_data.items():
        G.add_node(node)
        for child in data['children']:
            G.add_edge(node, child)
    
    # 计算布局
    pos = get_tree_layout(tree)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    
    # 准备节点颜色
    node_colors = []
    for node in G.nodes():
        if tree_data[node].get('pruned'):
            node_colors.append('#ff7f7f')  # 剪枝节点：红色
        elif node_types[node] is True:
            node_colors.append('#a6cee3')  # MAX节点：浅蓝
        elif node_types[node] is False:
            node_colors.append('#b2df8a')  # MIN节点：浅绿
        else:
            node_colors.append('#dddddd')  # 叶节点：浅灰
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
    
    # 准备节点标签
    labels = {}
    for node in G.nodes():
        data = tree_data[node]
        if data.get('pruned') and data['value'] == '?':
            label = f"{node}\n(已剪枝)"
        else:
            label = f"{node}\n值: {data['value']}"
            if node_types[node] is not None:  # 非叶节点显示α-β值
                label += f"\nα: {data['alpha']}\nβ: {data['beta']}"
        labels[node] = label
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', font_family='SimHei')
    
    plt.title("Alpha-Beta 剪枝博弈树", fontsize=16, fontfamily='SimHei', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图片到代码所在目录
    output_path = os.path.join(SCRIPT_DIR, 'alpha_beta_tree.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    return output_path

if __name__ == '__main__':
    # 运行Alpha-Beta剪枝
    tree_data = {}
    result = minimax_ab('A', True, float('-inf'), float('inf'), tree_data)
    print(f"博弈树的最优值: {result}")
    
    # 生成可视化
    output_path = visualize_tree(tree_data)
    print(f"博弈树已保存为: {output_path}")

    # 打印剪枝信息
    pruned_nodes = [node for node, data in tree_data.items() if data.get('pruned')]
    if pruned_nodes:
        print(f"被剪枝的节点: {', '.join(pruned_nodes)}")

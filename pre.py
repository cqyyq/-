import heapq
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import math
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GraphSearchVisualizer:
    def __init__(self, master):
        self.master = master
        master.title("图搜索算法可视化")
        
        # 创建图形对象
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 创建控制面板
        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 算法选择
        tk.Label(control_frame, text="选择算法:").pack(side=tk.LEFT)
        self.algorithm = tk.StringVar(value="DFS")
        algorithms = ["DFS", "BFS", "UCS", "A*", "Greedy"]
        algorithm_menu = ttk.Combobox(control_frame, textvariable=self.algorithm, values=algorithms)
        algorithm_menu.pack(side=tk.LEFT)
        
        # 起点和终点选择
        tk.Label(control_frame, text="start:").pack(side=tk.LEFT)
        self.start_node = tk.StringVar(value="Arad")
        start_menu = ttk.Combobox(control_frame, textvariable=self.start_node, values=list(graph.keys()))
        start_menu.pack(side=tk.LEFT)
        
        tk.Label(control_frame, text="destination:").pack(side=tk.LEFT)
        self.end_node = tk.StringVar(value="Bucharest")
        end_menu = ttk.Combobox(control_frame, textvariable=self.end_node, values=list(graph.keys()))
        end_menu.pack(side=tk.LEFT)
        
        # 开始按钮
        start_button = tk.Button(control_frame, text="开始搜索", command=self.start_search)
        start_button.pack(side=tk.LEFT, padx=10)
        
        # 重置按钮
        reset_button = tk.Button(control_frame, text="重置", command=self.reset)
        reset_button.pack(side=tk.LEFT)
        
        # 初始化图形
        self.G = nx.Graph()
        for node in graph:
            for neighbor, weight in graph[node]:
                self.G.add_edge(node, neighbor, weight=weight)
        
        # 固定布局
        self.pos = nx.spring_layout(self.G, seed=42)
        
        # 动画控制变量
        self.animation = None
        self.search_steps = []
        self.current_step = 0
        self.path = []
        self.total_cost = 0
        self.is_animating = False
        
        # 绘制初始图形
        self.draw_graph()
    
    def draw_graph(self, visited_nodes=None, current_path=None, current_node=None):
        """绘制图形当前状态"""
        self.ax.clear()
        
        # 绘制基本图形
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True,
                node_color="#ADD8E6", edge_color="gray", node_size=800, font_size=8)
        
        # 显示边权值
        edge_labels = {(u, v): d['weight'] for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=7, ax=self.ax)
        
        # 高亮已访问节点（如果有）
        if visited_nodes:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=visited_nodes,
                                 node_color="orange", node_size=800, ax=self.ax)
        
        # 高亮当前路径（如果有）
        if current_path and len(current_path) > 1:
            nx.draw_networkx_edges(self.G, self.pos,
                                  edgelist=[(current_path[i], current_path[i+1]) for i in range(len(current_path)-1)],
                                  edge_color="red", width=2, ax=self.ax)
        
        # 高亮当前节点（如果有）
        if current_node:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[current_node],
                                 node_color="red", node_size=800, ax=self.ax)
        
        # 显示标题信息
        title = f"{self.algorithm.get()} search: {self.start_node.get()} → {self.end_node.get()}"
        if current_path:
            title += f"\npath: {' → '.join(current_path)}"
        if self.total_cost > 0 and current_path and current_path[-1] == self.end_node.get():
            title += f"\ncost: {self.total_cost}"
        
        self.ax.set_title(title, pad=20, fontsize=10)
        self.canvas.draw()
    
    def heuristic(self, node, end):
        """计算启发式函数（欧几里得距离）"""
        pos_end = self.pos[end]
        pos_node = self.pos[node]
        return math.sqrt((pos_node[0]-pos_end[0])**2 + (pos_node[1]-pos_end[1])**2)
    
    def generate_search_steps(self):
        """根据选择的算法生成搜索步骤"""
        algorithm = self.algorithm.get()
        start = self.start_node.get()
        end = self.end_node.get()
        
        self.search_steps = []
        self.path = []
        self.total_cost = 0
        
        if algorithm == "DFS":
            self.dfs(start, end)
        elif algorithm == "BFS":
            self.bfs(start, end)
        elif algorithm == "UCS":
            self.ucs(start, end)
        elif algorithm == "A*":
            self.astar(start, end)
        elif algorithm == "Greedy":
            self.greedy(start, end)
    
    def dfs(self, start, end):
        """深度优先搜索"""
        stack = [(start, [start])]
        visited = set()
        
        while stack:
            node, path = stack.pop()
            self.search_steps.append((set(visited), path.copy(), node))
            
            if node == end:
                self.path = path
                self.total_cost = sum(graph[path[i]][0][1] for i in range(len(path)-1))
                return
            
            if node not in visited:
                visited.add(node)
                for neighbor, _ in graph.get(node, []):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
        
        self.search_steps.append((set(visited), [], ""))
    
    def bfs(self, start, end):
        """广度优先搜索"""
        queue = deque([(start, [start])])
        visited = set()
        
        while queue:
            node, path = queue.popleft()
            self.search_steps.append((set(visited), path.copy(), node))
            
            if node == end:
                self.path = path
                self.total_cost = sum(graph[path[i]][0][1] for i in range(len(path)-1))
                return
            
            if node not in visited:
                visited.add(node)
                for neighbor, _ in graph.get(node, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        self.search_steps.append((set(visited), [], ""))
    
    def ucs(self, start, end):
        """一致代价搜索"""
        heap = [(0, start, [start])]
        heapq.heapify(heap)
        visited = set()
        
        while heap:
            cost, node, path = heapq.heappop(heap)
            self.search_steps.append((set(visited), path.copy(), node))
            
            if node == end:
                self.path = path
                self.total_cost = cost
                return
            
            if node not in visited:
                visited.add(node)
                for neighbor, weight in graph.get(node, []):
                    if neighbor not in visited:
                        heapq.heappush(heap, (cost + weight, neighbor, path + [neighbor]))
        
        self.search_steps.append((set(visited), [], ""))
    
    def astar(self, start, end):
        """A*搜索"""
        heap = [(0 + self.heuristic(start, end), 0, start, [start])]
        heapq.heapify(heap)
        visited = {}
        
        while heap:
            _, g, node, path = heapq.heappop(heap)
            self.search_steps.append((set(visited.keys()), path.copy(), node))
            
            if node == end:
                self.path = path
                self.total_cost = g
                return
            
            if node not in visited or g < visited[node]:
                visited[node] = g
                for neighbor, weight in graph.get(node, []):
                    new_g = g + weight
                    if neighbor not in visited or new_g < visited.get(neighbor, float('inf')):
                        f_score = new_g + self.heuristic(neighbor, end)
                        heapq.heappush(heap, (f_score, new_g, neighbor, path + [neighbor]))
        
        self.search_steps.append((set(visited.keys()), [], ""))
    
    def greedy(self, start, end):
        """贪心搜索"""
        heap = [(self.heuristic(start, end), start, [start])]
        heapq.heapify(heap)
        visited = set()
        
        while heap:
            _, node, path = heapq.heappop(heap)
            self.search_steps.append((set(visited), path.copy(), node))
            
            if node == end:
                self.path = path
                self.total_cost = sum(graph[path[i]][0][1] for i in range(len(path)-1))
                return
            
            if node not in visited:
                visited.add(node)
                for neighbor, _ in graph.get(node, []):
                    if neighbor not in visited:
                        heapq.heappush(heap, (self.heuristic(neighbor, end), neighbor, path + [neighbor]))
        
        self.search_steps.append((set(visited), [], ""))
    
    def update_animation(self, frame):
        """更新动画帧"""
        if not self.is_animating:  # 如果重置被触发，停止动画
            return
        
        if frame < len(self.search_steps):
            visited, path, current_node = self.search_steps[frame]
            self.draw_graph(visited, path, current_node)
            self.current_step = frame
        else:
            self.is_animating = False
    
    def start_search(self):
        """开始搜索动画"""
        self.reset()  # 先重置确保状态干净
        
        self.is_animating = True
        self.generate_search_steps()
        self.current_step = 0
        
        # 创建动画
        self.animation = FuncAnimation(self.fig, self.update_animation,
                                     frames=len(self.search_steps),
                                     interval=800, repeat=False)
        self.canvas.draw()
    
    def reset(self):
        """重置图形"""
        self.is_animating = False  # 设置标志位停止动画
        
        if self.animation:
            try:
                self.animation.event_source.stop()
            except:
                pass
            self.animation = None
        
        # 重置所有状态变量
        self.search_steps = []
        self.current_step = 0
        self.path = []
        self.total_cost = 0
        
        # 清除图形并重新绘制初始状态
        self.ax.clear()
        self.draw_graph()
        self.canvas.draw()

# 图数据
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

# 创建主窗口并运行程序
root = tk.Tk()
app = GraphSearchVisualizer(root)
root.mainloop()
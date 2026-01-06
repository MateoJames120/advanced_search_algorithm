# Script demonstrating various advanced search algorithms in Pythonimport heapq
import time
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Set, Callable
from dataclasses import dataclass
from enum import Enum
import math

class SearchAlgorithm(Enum):
    BFS = "Breadth-First Search"
    DFS = "Depth-First Search"
    A_STAR = "A* Search"
    DIJKSTRA = "Dijkstra's Algorithm"
    BIDIRECTIONAL = "Bidirectional Search"
    BEST_FIRST = "Best-First Search"
    ITERATIVE_DEEPENING = "Iterative Deepening DFS"

@dataclass
class Node:
    """Node representation for graph/tree search"""
    state: any
    parent: Optional['Node'] = None
    path_cost: float = 0
    heuristic: float = 0
    depth: int = 0
    
    def __lt__(self, other: 'Node') -> bool:
        return (self.path_cost + self.heuristic) < (other.path_cost + other.heuristic)

class AdvancedSearch:
    def __init__(self, graph: Dict[any, Dict[any, float]] = None):
        self.graph = graph or {}
        self.nodes_expanded = 0
        self.search_time = 0
        
    def reset_stats(self):
        self.nodes_expanded = 0
        self.search_time = 0
    
    def bfs(self, start: any, goal: any) -> Tuple[List[any], Dict[str, int]]:
        """
        Breadth-First Search
        Returns: (path, statistics)
        """
        self.reset_stats()
        start_time = time.time()
        
        if start == goal:
            return [start], {"nodes_expanded": 0, "search_time": 0}
        
        frontier = deque([Node(start)])
        explored = set()
        frontier_set = {start}
        
        while frontier:
            current_node = frontier.popleft()
            frontier_set.remove(current_node.state)
            self.nodes_expanded += 1
            
            if current_node.state == goal:
                self.search_time = time.time() - start_time
                return self._reconstruct_path(current_node), {
                    "nodes_expanded": self.nodes_expanded,
                    "search_time": self.search_time
                }
            
            explored.add(current_node.state)
            
            for neighbor, cost in self.graph.get(current_node.state, {}).items():
                if neighbor not in explored and neighbor not in frontier_set:
                    child_node = Node(
                        state=neighbor,
                        parent=current_node,
                        path_cost=current_node.path_cost + cost,
                        depth=current_node.depth + 1
                    )
                    frontier.append(child_node)
                    frontier_set.add(neighbor)
        
        self.search_time = time.time() - start_time
        return None, {"nodes_expanded": self.nodes_expanded, "search_time": self.search_time}
    
    def dfs(self, start: any, goal: any, depth_limit: int = float('inf')) -> Tuple[Optional[List[any]], Dict[str, int]]:
        """
        Depth-First Search with optional depth limit
        """
        self.reset_stats()
        start_time = time.time()
        
        if start == goal:
            return [start], {"nodes_expanded": 0, "search_time": 0}
        
        frontier = [Node(start)]
        explored = set()
        
        while frontier:
            current_node = frontier.pop()
            self.nodes_expanded += 1
            
            if current_node.state == goal:
                self.search_time = time.time() - start_time
                return self._reconstruct_path(current_node), {
                    "nodes_expanded": self.nodes_expanded,
                    "search_time": self.search_time
                }
            
            explored.add(current_node.state)
            
            if current_node.depth < depth_limit:
                # Add neighbors in reverse order for more natural exploration
                neighbors = list(self.graph.get(current_node.state, {}).items())
                for neighbor, cost in reversed(neighbors):
                    if neighbor not in explored:
                        child_node = Node(
                            state=neighbor,
                            parent=current_node,
                            path_cost=current_node.path_cost + cost,
                            depth=current_node.depth + 1
                        )
                        frontier.append(child_node)
        
        self.search_time = time.time() - start_time
        return None, {"nodes_expanded": self.nodes_expanded, "search_time": self.search_time}
    
    def a_star_search(self, start: any, goal: any, 
                     heuristic_func: Callable[[any, any], float]) -> Tuple[Optional[List[any]], Dict[str, int]]:
        """
        A* Search Algorithm
        heuristic_func: function(state, goal) -> estimated cost
        """
        self.reset_stats()
        start_time = time.time()
        
        open_set = []
        heapq.heappush(open_set, Node(
            state=start,
            path_cost=0,
            heuristic=heuristic_func(start, goal)
        ))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic_func(start, goal)}
        open_set_hash = {start}
        
        while open_set:
            current_node = heapq.heappop(open_set)
            open_set_hash.remove(current_node.state)
            self.nodes_expanded += 1
            
            if current_node.state == goal:
                self.search_time = time.time() - start_time
                return self._reconstruct_path_from_dict(came_from, current_node.state), {
                    "nodes_expanded": self.nodes_expanded,
                    "search_time": self.search_time
                }
            
            for neighbor, cost in self.graph.get(current_node.state, {}).items():
                tentative_g_score = g_score[current_node.state] + cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node.state
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic_func(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, Node(
                            state=neighbor,
                            path_cost=tentative_g_score,
                            heuristic=heuristic_func(neighbor, goal)
                        ))
                        open_set_hash.add(neighbor)
        
        self.search_time = time.time() - start_time
        return None, {"nodes_expanded": self.nodes_expanded, "search_time": self.search_time}
    
    def dijkstra(self, start: any, goal: any) -> Tuple[Optional[List[any]], Dict[str, int]]:
        """
        Dijkstra's Algorithm - uniform cost search
        """
        self.reset_stats()
        start_time = time.time()
        
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        previous = {node: None for node in self.graph}
        
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            self.nodes_expanded += 1
            
            if current_node in visited:
                continue
                
            if current_node == goal:
                self.search_time = time.time() - start_time
                return self._reconstruct_path_from_dict(previous, current_node), {
                    "nodes_expanded": self.nodes_expanded,
                    "search_time": self.search_time
                }
            
            visited.add(current_node)
            
            for neighbor, weight in self.graph.get(current_node, {}).items():
                if neighbor in visited:
                    continue
                    
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        self.search_time = time.time() - start_time
        return None, {"nodes_expanded": self.nodes_expanded, "search_time": self.search_time}
    
    def bidirectional_search(self, start: any, goal: any) -> Tuple[Optional[List[any]], Dict[str, int]]:
        """
        Bidirectional Search
        """
        self.reset_stats()
        start_time = time.time()
        
        if start == goal:
            return [start], {"nodes_expanded": 0, "search_time": 0}
        
        # Forward search
        forward_frontier = deque([start])
        forward_parent = {start: None}
        
        # Backward search
        backward_frontier = deque([goal])
        backward_parent = {goal: None}
        
        intersection = None
        
        while forward_frontier and backward_frontier:
            # Expand forward
            current_forward = forward_frontier.popleft()
            self.nodes_expanded += 1
            
            for neighbor in self.graph.get(current_forward, {}):
                if neighbor not in forward_parent:
                    forward_parent[neighbor] = current_forward
                    forward_frontier.append(neighbor)
                    
                    if neighbor in backward_parent:
                        intersection = neighbor
                        break
            
            if intersection:
                break
                
            # Expand backward
            current_backward = backward_frontier.popleft()
            self.nodes_expanded += 1
            
            for neighbor in self.graph.get(current_backward, {}):
                if neighbor not in backward_parent:
                    backward_parent[neighbor] = current_backward
                    backward_frontier.append(neighbor)
                    
                    if neighbor in forward_parent:
                        intersection = neighbor
                        break
        
        if intersection:
            # Reconstruct path
            path = []
            
            # Forward part
            node = intersection
            while node is not None:
                path.append(node)
                node = forward_parent[node]
            path = path[::-1]
            
            # Backward part (excluding intersection which is already added)
            node = backward_parent[intersection]
            while node is not None:
                path.append(node)
                node = backward_parent[node]
            
            self.search_time = time.time() - start_time
            return path, {"nodes_expanded": self.nodes_expanded, "search_time": self.search_time}
        
        self.search_time = time.time() - start_time
        return None, {"nodes_expanded": self.nodes_expanded, "search_time": self.search_time}
    
    def iterative_deepening_dfs(self, start: any, goal: any, max_depth: int = 50) -> Tuple[Optional[List[any]], Dict[str, int]]:
        """
        Iterative Deepening Depth-First Search
        """
        self.reset_stats()
        start_time = time.time()
        
        for depth in range(max_depth + 1):
            path, stats = self.dfs(start, goal, depth_limit=depth)
            self.nodes_expanded += stats.get("nodes_expanded", 0)
            
            if path:
                self.search_time = time.time() - start_time
                return path, {
                    "nodes_expanded": self.nodes_expanded,
                    "search_time": self.search_time,
                    "depth_reached": depth
                }
        
        self.search_time = time.time() - start_time
        return None, {"nodes_expanded": self.nodes_expanded, "search_time": self.search_time}
    
    def best_first_search(self, start: any, goal: any, 
                         heuristic_func: Callable[[any, any], float]) -> Tuple[Optional[List[any]], Dict[str, int]]:
        """
        Greedy Best-First Search
        """
        self.reset_stats()
        start_time = time.time()
        
        frontier = []
        heapq.heappush(frontier, (heuristic_func(start, goal), Node(start)))
        explored = set()
        parent_map = {start: None}
        
        while frontier:
            _, current_node = heapq.heappop(frontier)
            self.nodes_expanded += 1
            
            if current_node.state == goal:
                self.search_time = time.time() - start_time
                return self._reconstruct_path_from_dict(parent_map, current_node.state), {
                    "nodes_expanded": self.nodes_expanded,
                    "search_time": self.search_time
                }
            
            explored.add(current_node.state)
            
            for neighbor, _ in self.graph.get(current_node.state, {}).items():
                if neighbor not in explored:
                    parent_map[neighbor] = current_node.state
                    heapq.heappush(frontier, (heuristic_func(neighbor, goal), Node(neighbor)))
        
        self.search_time = time.time() - start_time
        return None, {"nodes_expanded": self.nodes_expanded, "search_time": self.search_time}
    
    def _reconstruct_path(self, node: Node) -> List[any]:
        """Reconstruct path from goal node to start"""
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]
    
    def _reconstruct_path_from_dict(self, parent_map: Dict[any, any], goal: any) -> List[any]:
        """Reconstruct path using parent dictionary"""
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent_map.get(current)
        return path[::-1]

class SearchVisualizer:
    """Helper class for visualizing and comparing search algorithms"""
    
    @staticmethod
    def create_grid_graph(width: int, height: int) -> Dict[Tuple[int, int], Dict[Tuple[int, int], float]]:
        """Create a grid graph for testing"""
        graph = {}
        for x in range(width):
            for y in range(height):
                neighbors = {}
                # 4-directional movement
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbors[(nx, ny)] = 1.0
                graph[(x, y)] = neighbors
        return graph
    
    @staticmethod
    def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic for grid navigation"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    @staticmethod
    def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def benchmark_algorithms():
    """Benchmark and compare different search algorithms"""
    
    # Create a sample graph
    width, height = 10, 10
    graph = SearchVisualizer.create_grid_graph(width, height)
    
    start = (0, 0)
    goal = (9, 9)
    
    searcher = AdvancedSearch(graph)
    
    algorithms = {
        "BFS": lambda: searcher.bfs(start, goal),
        "DFS": lambda: searcher.dfs(start, goal),
        "A* (Manhattan)": lambda: searcher.a_star_search(start, goal, SearchVisualizer.manhattan_distance),
        "A* (Euclidean)": lambda: searcher.a_star_search(start, goal, SearchVisualizer.euclidean_distance),
        "Dijkstra": lambda: searcher.dijkstra(start, goal),
        "Bidirectional": lambda: searcher.bidirectional_search(start, goal),
        "Greedy Best-First": lambda: searcher.best_first_search(start, goal, SearchVisualizer.manhattan_distance),
        "Iterative Deepening": lambda: searcher.iterative_deepening_dfs(start, goal)
    }
    
    print("=" * 60)
    print("SEARCH ALGORITHM BENCHMARK")
    print("=" * 60)
    print(f"Graph: {width}x{height} grid")
    print(f"Start: {start}, Goal: {goal}")
    print("-" * 60)
    
    results = []
    
    for algo_name, algo_func in algorithms.items():
        print(f"\nRunning {algo_name}...")
        path, stats = algo_func()
        
        if path:
            path_length = len(path) - 1
            print(f"  ✓ Path found! Length: {path_length}")
        else:
            path_length = "N/A"
            print(f"  ✗ No path found")
        
        results.append({
            "Algorithm": algo_name,
            "Path Length": path_length,
            "Nodes Expanded": stats.get("nodes_expanded", 0),
            "Time (s)": f"{stats.get('search_time', 0):.6f}",
            "Success": path is not None
        })
    
    # Display comparison table
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'Path Length':<12} {'Nodes Expanded':<15} {'Time (s)':<12} {'Status':<10}")
    print("-" * 60)
    
    for result in results:
        status = "✓ Success" if result["Success"] else "✗ Failed"
        print(f"{result['Algorithm']:<20} {result['Path Length']:<12} "
              f"{result['Nodes Expanded']:<15} {result['Time (s)']:<12} {status:<10}")

def interactive_search():
    """Interactive demonstration of search algorithms"""
    
    # Example graph
    example_graph = {
        'A': {'B': 1, 'C': 4, 'D': 2},
        'B': {'A': 1, 'E': 3, 'F': 2},
        'C': {'A': 4, 'G': 5},
        'D': {'A': 2, 'H': 1},
        'E': {'B': 3, 'I': 2},
        'F': {'B': 2, 'J': 3, 'K': 4},
        'G': {'C': 5, 'L': 1},
        'H': {'D': 1, 'M': 3},
        'I': {'E': 2, 'N': 4},
        'J': {'F': 3, 'O': 2},
        'K': {'F': 4},
        'L': {'G': 1},
        'M': {'H': 3},
        'N': {'I': 4},
        'O': {'J': 2, 'P': 1},
        'P': {'O': 1}
    }
    
    def simple_heuristic(node, goal):
        # Simple heuristic based on alphabetical distance
        return abs(ord(node) - ord(goal))
    
    searcher = AdvancedSearch(example_graph)
    
    print("Interactive Search Demo")
    print("Graph nodes:", list(example_graph.keys()))
    
    while True:
        print("\nOptions:")
        print("1. BFS")
        print("2. DFS")
        print("3. A* Search")
        print("4. Dijkstra")
        print("5. Compare all")
        print("6. Exit")
        
        choice = input("\nSelect algorithm (1-6): ").strip()
        
        if choice == '6':
            break
        
        if choice not in ['1', '2', '3', '4', '5']:
            print("Invalid choice!")
            continue
        
        start = input("Enter start node: ").upper().strip()
        goal = input("Enter goal node: ").upper().strip()
        
        if start not in example_graph or goal not in example_graph:
            print("Invalid nodes!")
            continue
        
        if choice == '1':
            path, stats = searcher.bfs(start, goal)
            algo_name = "BFS"
        elif choice == '2':
            path, stats = searcher.dfs(start, goal)
            algo_name = "DFS"
        elif choice == '3':
            path, stats = searcher.a_star_search(start, goal, simple_heuristic)
            algo_name = "A*"
        elif choice == '4':
            path, stats = searcher.dijkstra(start, goal)
            algo_name = "Dijkstra"
        elif choice == '5':
            # Run all algorithms
            results = []
            for algo_func, name in [
                (lambda: searcher.bfs(start, goal), "BFS"),
                (lambda: searcher.dfs(start, goal), "DFS"),
                (lambda: searcher.a_star_search(start, goal, simple_heuristic), "A*"),
                (lambda: searcher.dijkstra(start, goal), "Dijkstra"),
                (lambda: searcher.bidirectional_search(start, goal), "Bidirectional"),
                (lambda: searcher.best_first_search(start, goal, simple_heuristic), "Best-First")
            ]:
                path, stats = algo_func()
                results.append((name, path, stats))
            
            print("\nComparison Results:")
            for name, path, stats in results:
                if path:
                    print(f"{name}: Path found ({len(path)-1} steps), "
                          f"Nodes expanded: {stats['nodes_expanded']}, "
                          f"Time: {stats['search_time']:.6f}s")
                else:
                    print(f"{name}: No path found")
            continue
        
        print(f"\n{algo_name} Results:")
        if path:
            print(f"Path: {' -> '.join(path)}")
            print(f"Path length: {len(path)-1} steps")
            print(f"Total cost: {stats.get('total_cost', 'N/A')}")
        else:
            print("No path found!")
        
        print(f"Nodes expanded: {stats['nodes_expanded']}")
        print(f"Search time: {stats['search_time']:.6f} seconds")

def main():
    """Main function"""
    print("Advanced Search Algorithms in Python")
    print("=" * 50)
    
    while True:
        print("\nSelect mode:")
        print("1. Benchmark algorithms (grid example)")
        print("2. Interactive search demo")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            benchmark_algorithms()
        elif choice == '2':
            interactive_search()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
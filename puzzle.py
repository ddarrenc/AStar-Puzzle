import sys
import time
from queue import PriorityQueue

class ASNode:
    def __init__(self, init_state, goal_state, weight, heuristic, g, depth):
        self.state = init_state
        self.goal = goal_state
        self.W = weight
        self.heuristic = heuristic
        self.path_cost = g
        self.depth = depth
        self.str = str_state(self.state)
        self.f = self.calculate_f()

    def __lt__(self, other):
        return self.f < other.f

    def goal_reached(self):
        for r in range(len(self.state)):
            for c in range(len(self.state[r])):
                if self.state[r][c] != self.goal[r][c]:
                    return False

        return True

    def expand(self):
        children = []
        i,j = find_value_in_state(self.state, 0)

        # Left
        if j != 0:
            move_left = [row[:] for row in self.state]
            move_left[i][j], move_left[i][j-1] = move_left[i][j-1], move_left[i][j]
            children.append(move_left)
        else:
            children.append(None)

        # Right
        if j != len(self.state[0]) - 1:
            move_right = [row[:] for row in self.state]
            move_right[i][j], move_right[i][j+1] = move_right[i][j+1], move_right[i][j]
            children.append(move_right)
        else:
            children.append(None)

        # Up
        if i != 0:
            move_up = [row[:] for row in self.state]
            move_up[i][j], move_up[i-1][j] = move_up[i-1][j], move_up[i][j]
            children.append(move_up)
        else:
            children.append(None)

        # Down
        if i != len(self.state) - 1:
            move_down = [row[:] for row in self.state]
            move_down[i][j], move_down[i+1][j] = move_down[i+1][j], move_down[i][j]
            children.append(move_down)
        else:
            children.append(None)

        return children

    def calculate_f(self):
        f = self.path_cost + (self.W * self.heuristic(self.state, self.goal))
        return f

def str_state(state):
    s = ""
    for r in range(len(state)):
        for c in range(len(state[r])):
            s += str(state[r][c]) + ' '

    return s

def build_state(values, grid_size):
    return [values[i:i+grid_size] for i in range(0, len(values), grid_size)]

def find_value_in_state(state, value):
    for i, row in enumerate(state):
        if value in row:
            return (i, row.index(value))
    
    raise ValueError("Could not find value within given state")

def sum_chessboard_distance(initial, goal):
    dist = 0

    for x1 in range(len(initial)):
        for y1 in range(len(initial[x1])):
            number = initial[x1][y1]
            if number != 0: # ignore empty square
                (x2,y2) = find_value_in_state(goal, number)
                dist += max(abs(x2-x1), abs(y2-y1))

    return dist


def setup():
    GRID_N = 4 
    n = len(sys.argv)

    if n != 2:
        raise TypeError("Incorrect number of arguments, pass only the input txt file")
    
    input_name = sys.argv[1]
    with open(input_name, 'r') as f:
        lines = f.readlines()
    
    if len(lines) != 11:
        raise ValueError("Incorrect input formatting, refer to documentation")

    weight = float(lines[0].strip())
    init_state_vals = []
    goal_state_vals = []

    for i in range(2, len(lines) - GRID_N - 1):
        init_str_vals = lines[i].strip().split(" ")
        goal_str_vals = lines[i + GRID_N + 1].strip().split(" ")
        for n in init_str_vals:
            init_state_vals.append(int(n))
        for m in goal_str_vals:
            goal_state_vals.append(int(m))

    
    init_state = build_state(init_state_vals, GRID_N)
    goal_state = build_state(goal_state_vals, GRID_N)

    solve(ASNode(init_state, goal_state, weight, sum_chessboard_distance, 0, 0))

def solve(root):
    # d = depth of shallowest goal
    # A = action list
    # F = f(n) values list
    # W = weight value of A*
    print(weighted_a_star(root))
   # d, A, F = weighted_a_star(pb)
   # W = pb.W


def weighted_a_star(root):
    pq = PriorityQueue()
    visited = set()
    generated = 0  # N
    moveset = ["L", "R", "U", "D"]
    visited.add(root.str)
    pq.put((root.f, root))
    generated += 1

    while not pq.empty():
        pair = pq.get()
        node = pair[1]
        visited.add(node.str)

        if node.goal_reached():
            return
        
        children = node.expand()

        for i, c in enumerate(children):
            if c is not None and str_state(c) not in visited:
                generated += 1
                child = ASNode(c, root.goal, root.W, root.heuristic, node.path_cost+1, node.depth+1)
                pq.put((child.f, child))
                visited.add(str_state(c))
                print(child.f)


    return None


if __name__ == "__main__":
    setup()

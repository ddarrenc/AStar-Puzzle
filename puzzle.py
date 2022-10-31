import sys
import copy
from queue import PriorityQueue

class ASNode:
    def __init__(self, grid_N, init_state, goal_state, weight, heuristic, path_cost, depth, coordinates, action=None):
        self.state = init_state
        self.goal = goal_state
        self.N = grid_N
        self.action = action

        self.W = weight
        self.heuristic = heuristic

        self.depth = depth
        self.g = path_cost

        self.blank_pos = coordinates
        self.f = self.g + (self.W * self.heuristic(self.state, self.goal, self.N))

    def legal_moves(self):
        legal = []
        i,j = self.blank_pos[0], self.blank_pos[1]

        # Left
        if j != 0:
            l_state = copy.deepcopy(self.state)
            l_state[i][j], l_state[i][j-1] = l_state[i][j-1], l_state[i][j]
            legal.append(ASNode(self.N, l_state, self.goal, self.W, self.heuristic, self.g+1, self.depth+1, (i,j-1), "L"))

        # Right
        if j != self.N - 1:
            r_state = copy.deepcopy(self.state)
            r_state[i][j], r_state[i][j+1] = r_state[i][j+1], r_state[i][j]
            legal.append(ASNode(self.N, r_state, self.goal, self.W, self.heuristic, self.g+1, self.depth+1, (i,j+1), "R"))

        # Up
        if i != 0:
            u_state = copy.deepcopy(self.state)
            u_state[i][j], u_state[i-1][j] = u_state[i-1][j], u_state[i][j]
            legal.append(ASNode(self.N, u_state, self.goal, self.W, self.heuristic, self.g+1, self.depth+1, (i-1,j), "U"))

        # Down
        if i != self.N - 1:
            d_state = copy.deepcopy(self.state)
            d_state[i][j], d_state[i+1][j] = d_state[i+1][j], d_state[i][j]
            legal.append(ASNode(self.N, d_state, self.goal, self.W, self.heuristic, self.g+1, self.depth+1, (i+1,j), "D"))

        return legal


def find_value_in_state(state, value, N):
    for i in range(N):
        for j in range(N):
            if value == state[i][j]:
                return (i,j)
    
    raise ValueError("Could not find value within given state")


def sum_chessboard_distance(initial, goal, N):
    dist = 0

    for x1 in range(N):
        for y1 in range(N):
            number = initial[x1][y1]
            if number != 0 and number != goal[x1][y1]: # ignore empty square and if already good
                (x2,y2) = find_value_in_state(goal, number, N)
                dist += max(abs(x2-x1) + abs(y2-y1))

    return dist


def sum_manhattan_distance(initial, goal, N):
    dist = 0

    for x1 in range(N):
        for y1 in range(N):
            number = initial[x1][y1]
            if number != 0 and number != goal[x1][y1]:
                (x2,y2) = find_value_in_state(goal, number, N)
                dist += abs(x2-x1) + abs(y2-y1)

    return dist


def setup():
    def build_state(values, grid_size):
        return [values[i:i+grid_size] for i in range(0, len(values), grid_size)]

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

    (x,y) = find_value_in_state(init_state, 0, GRID_N)
    root = ASNode(GRID_N, init_state, goal_state, weight, sum_chessboard_distance, 0, 0, (x,y))
    solve(root)

def solve(root):
    # d = depth of shallowest goal
    # A = action list
    # F = f(n) values list
    # W = weight value of A*

    # N=Nodes generated, d=depth, g=pathcost
    N, d, g = weighted_a_star(root)
    print(N, " ", d, " ", g)
   # d, A, F = weighted_a_star(pb)
   # W = pb.W

def toTuple(arr):
    return tuple(map(tuple, arr))

def weighted_a_star(root):
    pq = PriorityQueue()
    visited = set()
    generated = 1

    visited.add(toTuple(root.state))
    pq.put((root.f, id(root), root))

    while not pq.empty():
        node = pq.get()[2]
        visited.add(toTuple(node.state))
        print(node.f)

        if node.state == node.goal:
            return generated, node.depth, node.g

        children = node.legal_moves()

        for child in children:
            if toTuple(child.state) not in visited:
                generated += 1
                pq.put((child.f, id(child), child))
        


    return None


if __name__ == "__main__":
    setup()

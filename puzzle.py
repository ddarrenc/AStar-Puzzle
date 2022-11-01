import os
import time
import copy
from queue import PriorityQueue

class ASNode:
    def __init__(self, grid_N, init_state, goal_state, weight, heuristic, path_cost, depth, coordinates, prev=None, action=None):
        self.state = init_state
        self.goal = goal_state
        self.N = grid_N
        self.action = action

        self.W = weight
        self.heuristic = heuristic

        self.depth = depth
        self.g = path_cost

        self.blank_pos = coordinates
        self.prev = prev
        self.f = self.g + (self.W * self.heuristic(self.state, self.goal, self.N))

    def __lt__(self, other):
        return self.f < other.f

    def legal_moves(self):
        legal = []
        i,j = self.blank_pos[0], self.blank_pos[1]

        # Left
        if j != 0:
            l_state = copy.deepcopy(self.state)
            l_state[i][j], l_state[i][j-1] = l_state[i][j-1], l_state[i][j]
            legal.append(ASNode(self.N, l_state, self.goal, self.W, self.heuristic, self.g+1, self.depth+1, (i,j-1), self, "L"))

        # Right
        if j != self.N - 1:
            r_state = copy.deepcopy(self.state)
            r_state[i][j], r_state[i][j+1] = r_state[i][j+1], r_state[i][j]
            legal.append(ASNode(self.N, r_state, self.goal, self.W, self.heuristic, self.g+1, self.depth+1, (i,j+1), self, "R"))

        # Up
        if i != 0:
            u_state = copy.deepcopy(self.state)
            u_state[i][j], u_state[i-1][j] = u_state[i-1][j], u_state[i][j]
            legal.append(ASNode(self.N, u_state, self.goal, self.W, self.heuristic, self.g+1, self.depth+1, (i-1,j), self, "U"))

        # Down
        if i != self.N - 1:
            d_state = copy.deepcopy(self.state)
            d_state[i][j], d_state[i+1][j] = d_state[i+1][j], d_state[i][j]
            legal.append(ASNode(self.N, d_state, self.goal, self.W, self.heuristic, self.g+1, self.depth+1, (i+1,j), self, "D"))

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
                dist += max(abs(x2-x1), abs(y2-y1))

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


def solve():
    def build_state(values, grid_size):
        return [values[i:i+grid_size] for i in range(0, len(values), grid_size)]

    INPUT_PATH = "inputs/"
    OUTPUT_PATH = "outputs/"
    GRID_N = 4 
    
    
    tests_dir = [x for x in os.listdir(INPUT_PATH) if 'input' in x]

    for test in tests_dir:
        with open(INPUT_PATH + test, "r") as f:
            lines = f.readlines()

            if len(lines) != 11:
                raise ValueError("Incorrect input formatting for", test, "refer to documentation")

            weight_text = lines[0]
            init_text = ""
            goal_text = ""

            weight = float(weight_text.strip())
            init_state_vals = []
            goal_state_vals = []

            for i in range(2, len(lines) - GRID_N - 1):
                init_row = lines[i]
                goal_row = lines[i + GRID_N + 1]

                init_text += init_row
                goal_text += goal_row

                init_str_vals = init_row.strip().split(" ")
                goal_str_vals = goal_row.strip().split(" ")

                for n in init_str_vals:
                    init_state_vals.append(int(n))
                for m in goal_str_vals:
                    goal_state_vals.append(int(m))

            init_state = build_state(init_state_vals, GRID_N)
            goal_state = build_state(goal_state_vals, GRID_N)

            (x,y) = find_value_in_state(init_state, 0, GRID_N)
            root = ASNode(GRID_N, init_state, goal_state, weight, sum_chessboard_distance, 0, 0, (x,y))

            print("Attempting to solve", test + "...")
            start_time = time.time()
            print("Working...")
            d, N, A, F = weighted_a_star(root)
            end_time = time.time()

            output_suffix = test.split("input")[1]

            with open(OUTPUT_PATH + "output" + output_suffix, "w") as fo:
                fo.write(init_text + "\n" + goal_text.strip() + "\n\n" + weight_text + str(d) + "\n" + str(N)
                    + "\n" + A + "\n" + F)

            print("Solved", test, "in", str(end_time - start_time) + "s")
            print("Output written to", OUTPUT_PATH + "output" + output_suffix)
            print("***************************************************")
            print("Shallowest solution found at depth", d)
            print("Generated", N, "nodes")
            print("Actions:", A)
            print("***************************************************\n")

def toTuple(arr):
    return tuple(map(tuple, arr))

def weighted_a_star(root):
    pq = PriorityQueue()
    inQueue = set()
    visited = set()
    generated = 1

    rootTuple = toTuple(root.state)
    visited.add(rootTuple)
    inQueue.add(rootTuple)
    pq.put((root.f, root))

    while not pq.empty():
        node = pq.get()[1]
        nodeTuple = toTuple(node.state)
        visited.add(nodeTuple)
        inQueue.remove(nodeTuple)

        if node.state == node.goal:
            A = ""
            F = ""
            cursor = node
            while cursor is not None:
                if cursor.action is not None:
                    A = cursor.action + " " + A
                f = cursor.f
                if cursor.f % 1 == 0:
                    f = int(f)
                F = str(f) + " " + F
                cursor = cursor.prev

            return node.depth, generated, A, F

        children = node.legal_moves()
        generated += len(children)
        for child in children:
            childTuple = toTuple(child.state)
            if childTuple not in visited and childTuple not in inQueue:
                pq.put((child.f, child))
                inQueue.add(childTuple)

    return None

if __name__ == "__main__":
    solve()

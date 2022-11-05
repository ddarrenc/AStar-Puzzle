import os
import time
import copy
from queue import PriorityQueue

INPUT_PATH = "inputs/"    # Path of input text files
INPUT_SIZE = 11           # Required number of lines in input file
OUTPUT_PATH = "outputs/"  # Path of output text files
GRID_N = 4                # Size of puzzle board, 4 for 15-Puzzle


# Represents a single node for Weighted A Star Search
class ASNode:
    # Constructs a new ASNode
    #
    # grid_N: Size of the grid of which is NxN
    # init_state: NxN array representing the initial state of the puzzle
    # goal_state: NxN array representing the goal state of the puzzle
    # weight: Weight of the heuristic function
    # heuristic: The heuristic function to use
    # path_cost: Path cost up to this node
    # depth: Depth of the node
    # coordinates: (x,y) of the blank space
    # prev: Parent of this node, None for root
    # action: Action taken to go from parent to this node, None for root
    def __init__(self, grid_N, init_state, goal_state, weight, heuristic,
                 path_cost, depth, coordinates, prev=None, action=None):
        self.state = init_state  # initial state
        self.goal = goal_state   # goal state
        self.N = grid_N          # grid size
        self.action = action     # action to get to this node

        self.W = weight             # weight of heuristic
        self.heuristic = heuristic  # heuristic function

        self.depth = depth  # current depth
        self.g = path_cost  # path cost

        self.blank_pos = coordinates  # position of blank space
        self.prev = prev              # parent node

        # calculate f value using f(n) = g(n) + W*h(n)
        self.f = self.g + (self.W * self.heuristic(self.state,
                           self.goal, self.N))

    # Less than operator for PriorityQueue, based on f values of nodes
    def __lt__(self, other):
        return self.f < other.f

    # Returns a set of legal moves based on current state of the puzzle board
    def legal_moves(self):
        legal = []  # set of legal moves
        i, j = self.blank_pos[0], self.blank_pos[1]  # positions of blank space

        # Left
        if j != 0:
            l_state = copy.deepcopy(self.state)
            l_state[i][j], l_state[i][j-1] = l_state[i][j-1], l_state[i][j]
            legal.append(ASNode(self.N, l_state, self.goal, self.W,
                                self.heuristic, self.g+1, self.depth+1,
                                (i, j-1), self, "L"))

        # Right
        if j != self.N - 1:
            r_state = copy.deepcopy(self.state)
            r_state[i][j], r_state[i][j+1] = r_state[i][j+1], r_state[i][j]
            legal.append(ASNode(self.N, r_state, self.goal, self.W,
                                self.heuristic, self.g+1, self.depth+1,
                                (i, j+1), self, "R"))

        # Up
        if i != 0:
            u_state = copy.deepcopy(self.state)
            u_state[i][j], u_state[i-1][j] = u_state[i-1][j], u_state[i][j]
            legal.append(ASNode(self.N, u_state, self.goal, self.W,
                                self.heuristic, self.g+1, self.depth+1,
                                (i-1, j), self, "U"))

        # Down
        if i != self.N - 1:
            d_state = copy.deepcopy(self.state)
            d_state[i][j], d_state[i+1][j] = d_state[i+1][j], d_state[i][j]
            legal.append(ASNode(self.N, d_state, self.goal, self.W,
                                self.heuristic, self.g+1, self.depth+1,
                                (i+1, j), self, "D"))

        return legal


# Takes a 1D array and transforms into 2D array given the puzzle size
def build_state(values, grid_size):
    return [values[i:i+grid_size] for i in range(0, len(values), grid_size)]


# Transforms a 2D array to a 2D tuple, used for hashing in PriorityQueue
def toTuple(arr):
    return tuple(map(tuple, arr))


# Find index of a value within the current state
def find_value_in_state(state, value, N):
    for i in range(N):
        for j in range(N):
            if value == state[i][j]:
                return (i, j)

    raise ValueError("Could not find value within given state")


# Calculates the sum of chessboard distance from goal
def sum_chessboard_distance(initial, goal, N):
    dist = 0

    for x1 in range(N):
        for y1 in range(N):
            number = initial[x1][y1]
            # ignore empty square and if already good
            if number != 0 and number != goal[x1][y1]:
                (x2, y2) = find_value_in_state(goal, number, N)
                dist += max(abs(x2-x1), abs(y2-y1))

    return dist


# Calculates the sum of manhattan distance from goal
def sum_manhattan_distance(initial, goal, N):
    dist = 0

    for x1 in range(N):
        for y1 in range(N):
            number = initial[x1][y1]
            if number != 0 and number != goal[x1][y1]:
                (x2, y2) = find_value_in_state(goal, number, N)
                dist += abs(x2-x1) + abs(y2-y1)

    return dist


# Solve puzzles inside the inputs directory and writes solution into outputs
def solve():
    # Find all files with "input"
    tests_dir = sorted([x for x in os.listdir(INPUT_PATH) if "Input" in x])

    # Solve every input
    for test in tests_dir:
        with open(INPUT_PATH + test, "r") as f:
            lines = f.readlines()

            if len(lines) != INPUT_SIZE:
                raise ValueError("Incorrect input formatting for",
                                 test, "refer to documentation")

            weight_text = lines[0]
            init_text = ""
            goal_text = ""

            weight = float(weight_text.strip())
            init_state_vals = []
            goal_state_vals = []

            # Parse input file
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

            # Build the initial and goal states
            init_state = build_state(init_state_vals, GRID_N)
            goal_state = build_state(goal_state_vals, GRID_N)

            # Record position of blank space
            (x, y) = find_value_in_state(init_state, 0, GRID_N)

            # Construct the root node
            root = ASNode(GRID_N, init_state, goal_state,
                          weight, sum_chessboard_distance, 0, 0, (x, y))

            # Begin solving the puzzle board (assuming solvable)
            print("Attempting to solve", test + "...")
            start_time = time.time()
            print("Working...")
            # d = depth, N = generated, A = action list, F = f(n) value list
            d, N, A, F = weighted_a_star(root)  # Run search
            end_time = time.time()

            # Name of the output file
            output_suffix = test.split("Input")[1]

            # Write to the output
            with open(OUTPUT_PATH + "output" + output_suffix, "w") as fo:
                fo.write(init_text + "\n" + goal_text.strip() + "\n\n" +
                         weight_text + str(d) + "\n" + str(N)
                         + "\n" + A + "\n" + F)

            print("Solved", test, "in", str(end_time - start_time) + "s")
            print("Output written to", OUTPUT_PATH + "output" + output_suffix)
            print("***************************************************")
            #print(init_text)
            #print(goal_text, "\n")
            #print("Weight value:", weight_text.strip())
            print("Shallowest solution found at depth", d)
            print("Generated", N, "nodes")
            print("Actions:", A)
            #print("F-Values:", F)
            print("***************************************************\n")


# Weighted A Star
# Graph-Search (No repeated states)
def weighted_a_star(root):
    pq = PriorityQueue()
    inQueue = set()  # List of states currently in PriorityQueue
    visited = set()  # List of visited states
    generated = 1    # Number of nodes generated

    # Record that root has been visited and is in the PriorityQueue
    rootTuple = toTuple(root.state)
    visited.add(rootTuple)
    inQueue.add(rootTuple)
    pq.put((root.f, root))

    # As long as there are still valid states, keep going
    while not pq.empty():
        node = pq.get()[1]  # Get node from queue
        nodeTuple = toTuple(node.state)
        visited.add(nodeTuple)  # Record node has been visited
        inQueue.remove(nodeTuple)  # Remove node from queue tracker

        # Check if the goal state has been reached
        if node.state == node.goal:
            A = ""
            F = ""
            cursor = node

            # Reconstruct the actions and f(n) values from root to goal
            while cursor is not None:
                if cursor.action is not None:
                    A = cursor.action + " " + A
                f = cursor.f
                if cursor.f % 1 == 0:
                    f = int(f)
                F = str(f) + " " + F
                cursor = cursor.prev

            # Return the depth of the solution, number of nodes generated,
            # action list, and f(n) values list
            return node.depth, generated, A, F

        # Generate all legal moves from this state
        children = node.legal_moves()

        # For each child, if not visited and not already in the queue, add it
        for child in children:
            childTuple = toTuple(child.state)
            if childTuple not in visited and childTuple not in inQueue:
                generated += 1
                pq.put((child.f, child))
                inQueue.add(childTuple)

    # No solution found
    return -1, generated, None, None


if __name__ == "__main__":
    solve()

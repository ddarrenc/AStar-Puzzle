from curses import KEY_SMESSAGE
import sys

class PuzzleBoard:
    def __init__(self, init_state, goal_state, weight):
        self.curr = init_state
        self.goal = goal_state
        self.W = weight
        self.path_cost = 0
        self.f = self.path_cost + self.heur()

    def heur(self):
        return 0
        

def run_search(lines):
    weight = float(lines[0].strip())
    init_state = []
    goal_state = []

    for i in range(2, len(lines) - GRID_N - 1):
        init_str_vals = lines[i].strip().split(" ")
        goal_str_vals = lines[i + GRID_N + 1].strip().split(" ")
        for n in init_str_vals:
            init_state.append(int(n))
        for m in goal_str_vals:
            goal_state.append(int(m))
    
    pb = PuzzleBoard(init_state, goal_state, weight)


if __name__ == "__main__":
    GRID_N = 4 

    n = len(sys.argv)

    if n != 2:
        raise TypeError("Incorrect number of arguments, pass only the input txt file")
    
    input_name = sys.argv[1]

    with open(input_name, 'r') as f:
        lines = f.readlines()
    
    if len(lines) != 11:
        raise ValueError("Incorrect input formatting, refer to documentation")

    run_search(lines)
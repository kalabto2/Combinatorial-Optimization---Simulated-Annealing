import argparse
import random
import copy
import math

parser = argparse.ArgumentParser(description='Compute SAT problem with Simulated Annealing.')
parser.add_argument('filepath', type=str, help='filepath to instance')
parser.add_argument('-t', '--init_temperature', type=int, help='Initial temperature of Simulated Annealing', default=10000)
parser.add_argument('-f', '--frozen_boundary', type=int, help='temperature, where algorithm freeze', default=100)
parser.add_argument('-e', '--equilibrium', type=int, help='steps between cooling', default=250)
parser.add_argument('-c', '--cooling', type=float, help='Cooling constant', default=0.95)
parser.add_argument('-w', '--white_box', action='store_true', help="Prints verbose")
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
args = parser.parse_args()


def sat_clause(state: [bool], clause: [int]):
    for i in clause:
        if state[abs(i) - 1] == (True if i > 0 else False):
            return True
    return False


class SimulatedAnnealing:
    def __init__(self, filepath: str):
        f = open(filepath, "r")
        f_content = f.read()
        lines = f_content.split("\n")

        self.CLAUSES: [[int]] = []
        for line in lines[:-1]:
            # skip comments or unnecessary lines
            if line[0] == 'c' or line[0] == 'p':
                continue

            # load weights of variables
            if line[0] == 'w':
                self.WEIGHTS: [int] = list(map(int, line.split(' ')[1:-1]))
                continue

            # load clauses
            self.CLAUSES.append(list(map(int, line.strip().split(' ')[:-1])))

        self.C_CLAUSES: int = len(self.CLAUSES)
        self.C_VARIABLES: int = len(self.WEIGHTS)

    def count_weights(self, state: [bool]):
        total = 0
        for (on, w) in zip(state, self.WEIGHTS):
            if on:
                total += w
        return total

    def sat_clauses(self, state: [bool]):
        sat_cl = []
        for clause in self.CLAUSES:
            sat_cl.append(sat_clause(state, clause))
        return sat_cl

    def count_of_sat_clauses(self, state: [bool]):
        return self.sat_clauses(state).count(True)

    def is_sat(self, state: [bool]):
        return self.count_of_sat_clauses(state) == self.C_CLAUSES

    def get_random_state(self):
        return [bool(random.randrange(0, 2)) for _ in range(self.C_VARIABLES)]

    def frozen(self, temperature: float) -> bool:
        return temperature <= args.frozen_boundary

    def equilibrium(self):
        return args.equilibrium

    def cool(self, temperature: float):
        return temperature * args.cooling

    def random_bitflip(self, state: [bool]):
        new_state = copy.deepcopy(state)
        idx = random.randrange(0, self.C_VARIABLES)
        new_state[idx] = not new_state[idx]
        return new_state

    def get_neighbor(self, state: [bool]):
        return self.random_bitflip(state)

    def cost(self, state: [bool]):
        return self.count_weights(state) * self.count_of_sat_clauses(state) / self.C_CLAUSES

    def new_is_better(self, old_state: [bool], new_state: [bool]):
        return self.cost(new_state) > self.cost(old_state)

    def how_much_worse(self, old_state: [bool], new_state: [bool]):
        return self.cost(new_state) - self.cost(old_state)

    def new_state(self, old_state: [bool], temperature: float):
        new_state = self.get_neighbor(old_state)

        if self.new_is_better(old_state, new_state):
            return new_state

        # if self.too_heavy(new_state):
        #     return old_state

        delta = self.how_much_worse(old_state, new_state)
        if random.uniform(0, 1) < math.exp(-delta / temperature):
            return new_state

        return old_state

    def compute(self):
        # get init temperature
        temperature: float = args.init_temperature

        # get init state
        state = self.get_random_state()
        best = copy.deepcopy(state)

        i = 0
        while not self.frozen(temperature):
            # print(">", i)
            j = 0
            while j < self.equilibrium():
                state = self.new_state(state, temperature)

                # self.sa_cost_per_iteration.append(self.count_total_price(state))
                # print(">>", j, "W:", self.count_total_weight(state), "\t\tP:", self.count_total_price(state))
                if self.new_is_better(best, state):
                    best = copy.deepcopy(state)
                    # print(">>> W:", self.count_total_weight(best), "\t\tP:", self.count_total_price(best))
                j += 1
            temperature = self.cool(temperature)
            i += 1

        return best


if __name__ == "__main__":
    solver = SimulatedAnnealing(args.filepath)
    solution = solver.compute()
    print(solver.count_weights(solution), solver.count_of_sat_clauses(solution), solver.C_CLAUSES)

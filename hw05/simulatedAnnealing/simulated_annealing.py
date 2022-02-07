import argparse
import random
import copy
import math
from time import time
import re

parser = argparse.ArgumentParser(description='Compute SAT problem with Simulated Annealing.')
parser.add_argument('filepath', type=str, help='filepath to instance')
parser.add_argument('-t', '--init_temperature', type=int, help='Initial temperature of Simulated Annealing. Default '
                                                               'is feedback control configuration.', default=None)
parser.add_argument('-f', '--frozen_boundary', type=int, help='temperature, where algorithm freeze. Default is '
                                                              'freezing after superiority of the most valuable state '
                                                              'in last 100 states.', default=None)
parser.add_argument('-e', '--equilibrium', type=int, help='steps between cooling. Default is equilibrium auto '
                                                          'computation.', default=None)
parser.add_argument('-c', '--cooling', type=float, help='Cooling constant. Default is 0.7 .', default=0.7)
parser.add_argument('-w', '--white_box', action='store_true', help="Prints additional data.")
args = parser.parse_args()


def sat_clause(state: [bool], clause: [int]):
    for i in clause:
        if state[abs(i) - 1] == (True if i > 0 else False):
            return True
    return False


class SimulatedAnnealing:
    def __init__(self, filepath: str):
        filename_parts = filepath.split("/")[-1].split("-")
        self.ID = "-".join(filename_parts[:2])
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
                line2 = re.sub(" +", " ", line)
                self.WEIGHTS: [int] = list(map(int, line2.split(' ')[1:-1]))
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

    def smart_frozen(self, temperature: float, cost_per_iteration: [float]) -> bool:
        if len(cost_per_iteration) == 0:
            return False
        c = 100
        return cost_per_iteration[-c:].count(max(cost_per_iteration[-c:])) >= c/2
        # return len(set(cost_per_iteration[-c:])) == 1

    def frozen(self, temperature: float, cost_per_iteration: [float]) -> bool:
        if args.frozen_boundary is None:
            return self.smart_frozen(temperature, cost_per_iteration)
        return temperature <= args.frozen_boundary

    def equilibrium(self):
        if args.equilibrium is None:
            return self.C_VARIABLES * 10
        return args.equilibrium

    def cool(self, temperature: float):
        return temperature * args.cooling

    # --------------- Neighbor seeking ---------------

    def random_bitflip(self, state: [bool]):
        new_state = copy.deepcopy(state)
        idx = random.randrange(0, self.C_VARIABLES)
        new_state[idx] = not new_state[idx]
        return new_state

    def random_unsat_clause_bitflip(self, state: [bool]):
        new_state = copy.deepcopy(state)
        clause_idx = random.randrange(0, self.sat_clauses(state).count(False))
        var_in_clause_idx = random.randrange(0, len(self.CLAUSES[clause_idx]))
        var_idx = abs(self.CLAUSES[clause_idx][var_in_clause_idx]) - 1
        new_state[var_idx] = not new_state[var_idx]
        return new_state

    def get_neighbor(self, state: [bool]):
        return self.random_bitflip(state)
        # if random.randrange(0, 100) == 0:
        #     return self.get_random_state()
        # if self.is_sat(state):
        #     return self.random_bitflip(state)
        # return self.random_unsat_clause_bitflip(state)

    # --------------- State cost ---------------

    def new_is_best(self, old_state: [bool], new_state: [bool]):
        old_sat_rate = self.count_of_sat_clauses(old_state) / float(self.C_CLAUSES)
        old_weight = self.count_weights(old_state)
        new_sat_rate = self.count_of_sat_clauses(new_state) / float(self.C_CLAUSES)
        new_weight = self.count_weights(new_state)
        if new_sat_rate > old_sat_rate:
            return True
        elif new_sat_rate == old_sat_rate:
            return new_weight > old_weight
        return False

    def cost(self, state: [bool]):
        # commented code below are older versions of cost function
        # return self.count_weights(state) / (2 ** (self.C_CLAUSES - self.count_of_sat_clauses(state)))
        # return self.count_weights(state) * (self.count_of_sat_clauses(state) / self.C_CLAUSES) ** 2
        return self.count_weights(state) / ((self.C_CLAUSES - self.count_of_sat_clauses(state) ) ** 2 + 1)
        # return self.count_weights(state) * (self.count_of_sat_clauses(state) / self.C_CLAUSES)
        # return self.count_weights(state) * (self.count_of_sat_clauses(state) / self.C_CLAUSES)

    def new_is_better(self, old_state: [bool], new_state: [bool]):
        return self.cost(new_state) > self.cost(old_state)

    def how_much_worse(self, old_state: [bool], new_state: [bool]):
        return self.cost(old_state) - self.cost(new_state)

    def new_state(self, old_state: [bool], temperature: float):
        new_state = self.get_neighbor(old_state)

        if self.new_is_better(old_state, new_state):
            return new_state

        delta = self.how_much_worse(old_state, new_state)
        if random.uniform(0, 1) < math.exp(-delta / temperature):
            return new_state

        return old_state

    def create_output_string(self, state: [bool]):
        adjusted_state = []
        for i, up in zip(range(self.C_VARIABLES), state):
            adjusted_state.append(str(i + 1) if up else str(-(i + 1)))
        adjusted_state.append("0")
        return self.ID[1:] + " " + str(self.count_weights(state)) + " " + " ".join(adjusted_state)

    def compute(self, white_box=False):
        # get init temperature
        temperature: float = args.init_temperature

        # get init state
        state = self.get_random_state()
        best = copy.deepcopy(state)

        i = 0
        cost_per_iteration = []
        sat_clause_rate_per_iteration = []
        while not self.frozen(temperature, cost_per_iteration):
            j = 0
            while j < self.equilibrium():
                state = self.new_state(state, temperature)

                cost_per_iteration.append(self.cost(state))
                if white_box:
                    sat_clause_rate_per_iteration.append(self.count_of_sat_clauses(state) / self.C_CLAUSES)

                if self.new_is_best(best, state):
                    best = copy.deepcopy(state)

                j += 1
            temperature = self.cool(temperature)
            i += 1

        if white_box:
            return best, cost_per_iteration, sat_clause_rate_per_iteration
        return best

    # --------------- Initial temperature ---------------

    def get_total_weights(self):
        return self.count_weights([True] * self.C_VARIABLES)

    def get_worse_state_prob_acceptance(self, temperature: float):
        # get init state
        state = self.get_random_state()

        j = 0
        worse_accepted = 0
        while j < 20:
            new_state = self.new_state(state, temperature)
            if self.cost(new_state) < self.cost(state):  # not self.new_is_better(state, new_state):
                worse_accepted += 1
            state = new_state
            j += 1

        return worse_accepted / float(j)

    def feedback_control(self):
        init_temperature: float = self.get_total_weights() // self.C_VARIABLES
        prob_boundary = 0.5

        while True:
            # get probability of worse state acceptance
            worse_state_prob_acceptance = self.get_worse_state_prob_acceptance(init_temperature)

            # if probability of worse state acceptance is enough high, break
            if worse_state_prob_acceptance >= prob_boundary:
                return init_temperature

            # else increase temperature
            init_temperature *= 2


if __name__ == "__main__":
    solver = SimulatedAnnealing(args.filepath)

    # set initial_temperature if not set constant
    if args.init_temperature is None:
        args.init_temperature = solver.feedback_control()

    # calculate
    t1 = time()
    if args.white_box:
        solution, cost_per_iteration, sat_clause_rate_per_iteration = solver.compute(white_box=args.white_box)
    else:
        solution = solver.compute(white_box=args.white_box)
    t2 = time()

    # print data on output
    print(solver.create_output_string(solution))
    print(solver.count_weights(solution),
          solver.count_of_sat_clauses(solution),
          solver.C_CLAUSES,
          f"{(t2-t1):.2f}",
          solver.ID[1:])
    if args.white_box:
        print(" ".join(list(map(str, cost_per_iteration))))
        print(" ".join(list(map(str, sat_clause_rate_per_iteration))))

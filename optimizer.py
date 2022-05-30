import numpy as np
import tsp

class RandomSearch:

    def __init__(self, N: int, MAX_EVALUATIONS: int, PROB_DIMENSION: int):
        """is called when generating this class

        Args:
            N (int): number of solutions
            MAX_EVALUATIONS (int): number of maximum evaluations
            PROB_DIMENSION (int): number of the cities of TSP
        """
        # RS settings
        self.N = N

        # Problem settings
        self.MAX_EVALUATIONS = MAX_EVALUATIONS
        self.PROB_DIMENSION = PROB_DIMENSION

        # Private variables
        self.Xs = [None for _ in range(self.N)]
        self.Fs = [None for _ in range(self.N)]
        self.BestX = None
        self.BestFX = None
        self.BASICTOUR = [i for i in range(0, self.PROB_DIMENSION)]  # [0,1,2,...,50]

    def initialization(self):
        """is called to randomly initialize all solutions
        """
        for i in range(self.N):
            self.Xs[i] = np.random.permutation(self.BASICTOUR)

    def evaluate(self, tsp: tsp.TSP):
        """calculates fitness (Fs) of each solution (Xs) for TSP

        Args:
            tsp (tsp.TSP): class of TSP
        """
        for i in range(self.N):
            self.Fs[i] = tsp.evaluate(self.Xs[i])

    def update(self):
        """updates the best solution (BestX) and fitness (BestFX) if a better solution appears
        """
        _minID = np.argmin(self.Fs)
        if self.BestFX == None or self.Fs[_minID] < self.BestFX:
            self.BestX, self.BestFX = self.Xs[_minID], self.Fs[_minID]

    def generation(self):
        """same as self.initialization; completely randomly generates next generation
        """
        self.initialization()

class GeneticAlgorithm:

    def __init__(self, N: int, CrossoverRate: float, MutationRate: float, TournamentSize: int, MAX_EVALUATIONS: int, PROB_DIMENSION: int):
        """is called when generating this class

        Args:
            N (int): number of solutions
            CrossoverRate (float): possibility of occurrence of crossover
            MutationRate (float): possibility of occurrence of mutation
            TournamentSize (int): number of candidates of tournament selection
            MAX_EVALUATIONS (int): number of maximum evaluations
            PROB_DIMENSION (int): number of the cities of TSP
        """
        # GA settings
        self.N = N
        self.PC = CrossoverRate
        self.PM = MutationRate
        self.TS = TournamentSize

        # Problem settings
        self.MAX_EVALUATIONS = MAX_EVALUATIONS
        self.PROB_DIMENSION = PROB_DIMENSION

        # Private variables
        self.Xs = [None for _ in range(self.N)]
        self.Fs = [None for _ in range(self.N)]
        self.BestX = None
        self.BestFX = None
        self.BASICTOUR = [i for i in range(0, self.PROB_DIMENSION)]  # [0,1,2,...,50]

    def initialization(self):
        """is called only one time to initialize all solutions when starting GA
        """
        for i in range(self.N):
            self.Xs[i] = np.random.permutation(self.BASICTOUR)

    def evaluate(self, tsp: tsp.TSP):
        """calculates fitness (Fs) of each solution (Xs) for TSP

        Args:
            tsp (tsp.TSP): class of TSP
        """
        for i in range(self.N):
            self.Fs[i] = tsp.evaluate(self.Xs[i])

    def update(self):
        """updates the best solution (BestX) and fitness (BestFX) if a better solution appears
        """
        _minID = np.argmin(self.Fs)
        if self.BestFX == None or self.Fs[_minID] < self.BestFX:
            self.BestX, self.BestFX = self.Xs[_minID], self.Fs[_minID]

    def generation(self):
        """consecutive procedure of selection, crossover and mutation to generate two offsprings from parents
        """
        nextXs = []
        for _ in range((int)(self.N/2)):
            #offspring1, offspring2 = self.crossover(self.tournamentSelection(), self.tournamentSelection())
            offspring1, offspring2 = self.crossover(self.rouletteWheelSelection(), self.rouletteWheelSelection())
            nextXs.append(self.mutation(offspring1))
            nextXs.append(self.mutation(offspring2))
        self.Xs = nextXs

    def tournamentSelection(self):
        """selects one solution being based on tournamentSelection

        Returns:
            list: selected solution
        """
        # generate candidates for the tournament out of all solutions
        candidates = np.random.choice([Individual(self.Xs[i], self.Fs[i]) for i in range(self.N)], self.TS, False)
        # search an index of the solution whose fitness is minimum
        min_fitness_candidateID = np.argmin([i.Fs for i in candidates])
        return candidates[min_fitness_candidateID].Xs

    def rouletteWheelSelection(self):
        """selects one solution being based on rouletteWheelSelection

        Returns:
            list: selected solution
        """
        Fs_sum  = sum(self.Fs)
        # normalize self.Fs to use itself as possibility distribution
        Fs_normalize = [i / Fs_sum for i in self.Fs]
        # select one solution out of all solutions according to the possibility distribution Fs_normalize
        selected_Xs = np.random.choice([Individual(self.Xs[i], self.Fs[i]) for i in range(self.N)], 1, Fs_normalize)
        return selected_Xs[0].Xs

    def crossover(self, parent1: list, parent2 :list):
        """does crossover between two solutions

        Args:
            parent1 (list): father solution
            parent2 (list): mother solution

        Returns:
            tuple: consists of two lists of offspring solutions
        """
        if np.random.rand() < self.PC:
            # randomly select two slice point
            slice_point = np.random.choice([i for i in range(0, self.PROB_DIMENSION + 1)], 2, False)
            slice_point = sorted(slice_point)

            # slice from parent1
            parent1_sliced = list(parent1[slice_point[0]:slice_point[1]])

            # rest of parent2
            parent2_rest = [i for i in parent2 if i not in parent1_sliced]

            # slice from parent2
            parent2_sliced = list(parent2[slice_point[0]:slice_point[1]])

            # rest of parent1
            parent1_rest = [i for i in parent1 if i not in parent2_sliced]

            # generate two offsprings
            offspring1, offspring2 = parent2_rest[:slice_point[0]] + parent1_sliced + parent2_rest[slice_point[0]:], parent1_rest[:slice_point[0]] + parent2_sliced + parent1_rest[slice_point[0]:]
        else:
            offspring1, offspring2 = parent1, parent2
        return offspring1, offspring2

    def mutation(self, offspring: list):
        """does mutation

        Args:
            offspring (list): solution

        Returns:
            list: mutated solution
        """
        if np.random.rand() < self.PM:
            # randomly select two indexes
            mutate_indexes = np.random.choice([i for i in range(0, self.PROB_DIMENSION)], 2, False)
            # swap the two
            offspring[mutate_indexes[0]], offspring[mutate_indexes[1]] = offspring[mutate_indexes[1]], offspring[mutate_indexes[0]]
        else:
            pass
        return offspring

class Individual:
    """class of an individual solution whose member is Xs (solution) and Fs (fitness)
    """
    def __init__(self,  Xs, Fs):
        self.Xs = Xs
        self.Fs = Fs


def run(problem: tsp.TSP, optimizer: GeneticAlgorithm, MAX_EVALUATIONS: int, filename: str):
    """solve a problem

    Args:
        problem (tsp.TSP): a problem
        optimizer (GeneticAlgorithm): an optimizer
        MAX_EVALUATIONS (int): number of maximum evaluations
        filename (str): name of algorithm
    """
    print("run {}".format(filename))

    evals = 0
    log = []

    optimizer.initialization()
    optimizer.evaluate(problem)

    while evals < MAX_EVALUATIONS:
        optimizer.generation()
        optimizer.evaluate(problem)
        optimizer.update()
        evals += optimizer.N

        # logging
        print(evals, round(optimizer.BestFX, 1))
        log.append([evals, round(optimizer.BestFX, 1)])
    np.savetxt('_out_{}.csv'.format(filename), log, delimiter=',')


if __name__ == "__main__":
    # Basic setting (Do NOT change)
    N, MAX_EVALUATIONS, PROB_DIMENSION = 100, 50000, 50
    TSP = tsp.TSP(PROB_DIMENSION)

    # run Random search
    RS = RandomSearch(N, MAX_EVALUATIONS, PROB_DIMENSION)
    run(TSP, RS, MAX_EVALUATIONS, "RS")

    # run Genetic algorithm
    CrossoverRate, MutationRate, TournamentSize = 1.0, 0.1, 10
    GA = GeneticAlgorithm(N, CrossoverRate, MutationRate, TournamentSize, MAX_EVALUATIONS, PROB_DIMENSION)
    run(TSP, GA, MAX_EVALUATIONS, "GA")

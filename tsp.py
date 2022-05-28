import numpy as np
import math

class TSP:

    def __init__(self, PROB_DIMEINTION):
        self.PROB_DIMEINTION = PROB_DIMEINTION
        self.rd = np.random
        self.rd.seed(1)

        self.map_info = self.TSP_mapGenerator()

    def TSP_mapGenerator(self):
        x_coord  = [self.rd.randint(0, 100) for _ in range(self.PROB_DIMEINTION)]
        y_coord  = [self.rd.randint(0, 100) for _ in range(self.PROB_DIMEINTION)]

        coord    = [[x_coord[i], y_coord[i]] for i in range(self.PROB_DIMEINTION)]
        pos_info = []
        for i in range(self.PROB_DIMEINTION):
            pos_info.append(coord[i])
        return pos_info

    def evaluate(self, x):
        if not len(x) == self.PROB_DIMEINTION:
            print("Error: Solution X is not a {}-d vector".format(self.PROB_DIMEINTION))
            return None

        _dst  = 0
        for i in range(len(x)-1):
            _prev = self.map_info[x[i]]
            _next = self.map_info[x[i+1]]

            _dst += math.sqrt(math.pow(_prev[0]-_next[0], 2) + math.pow(_prev[1]-_next[1], 2))

        return _dst
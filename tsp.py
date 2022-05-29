import numpy as np
import math

class TSP:

    def __init__(self, PROB_DIMEINTION: int):
        """is called when making a TSP problem

        Args:
            PROB_DIMEINTION (int): number of cities
        """
        self.PROB_DIMEINTION = PROB_DIMEINTION
        self.rd = np.random
        self.rd.seed(1)

        self.map_info = self.TSP_mapGenerator()

    def TSP_mapGenerator(self):
        """generates a map of cities as (x, y) randomly between [0, 100)

        Returns:
            list: location (i.e. combination of (x, y)) of cities
        """
        # randomly location of the city as (x, y)
        x_coord  = [self.rd.randint(0, 100) for _ in range(self.PROB_DIMEINTION)]
        y_coord  = [self.rd.randint(0, 100) for _ in range(self.PROB_DIMEINTION)]

        coord    = [[x_coord[i], y_coord[i]] for i in range(self.PROB_DIMEINTION)]

        # pos_info is idential to coord
        pos_info = []
        for i in range(self.PROB_DIMEINTION):
            pos_info.append(coord[i])
        return pos_info

    def evaluate(self, x: list):
        """calculate fitness out of a solution

        Args:
            x (list): solution; self.PROB_DIMEINTION-d vector

        Returns:
            float: sum of the tarvel distance
        """
        if not len(x) == self.PROB_DIMEINTION:
            print("Error: Solution X is not a {}-d vector".format(self.PROB_DIMEINTION))
            return None

        # iterately adds the distance between adjacent two cities
        _dst  = 0
        for i in range(len(x)-1):
            _prev = self.map_info[x[i]]
            _next = self.map_info[x[i+1]]

            _dst += math.sqrt(math.pow(_prev[0]-_next[0], 2) + math.pow(_prev[1]-_next[1], 2))

        return _dst

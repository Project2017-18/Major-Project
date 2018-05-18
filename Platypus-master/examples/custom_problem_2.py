from platypus import NSGAII, OMOPSO,Problem, Real,Binary
import numpy as np


class Schaffer(Problem):

    def __init__(self):
        super(Schaffer, self).__init__(1, 2)
       # self.types[:] = Binary(5)
        self.types[:] = Real(-10,10)
    def evaluate(self, solution):
        x = solution.variables[:]
        print("x")
        print(x)
        solution.objectives[:] = [x[0], (x[0]-2)**4]
algorithm = OMOPSO(Schaffer(),None)
algorithm.run(100)
# plot the results using matplotlib
import matplotlib.pyplot as plt

plt.scatter([s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])

plt.xlabel("$f_1(x)$")
plt.ylabel("$f_2(x)$")
plt.show()

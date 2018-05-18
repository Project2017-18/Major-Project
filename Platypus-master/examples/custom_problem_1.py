from platypus import NSGAII, OMOPSO,Problem, Real
import numpy as np
def schaffer(x):
    print("x")
    print(x)
    print("x[0]")
    print(x[0])
    return [x[0]**2, (x[0]-2)**2]

problem = Problem(1, 2)
problem.types[:] = Real(0, 1)
problem.function = schaffer

algorithm = NSGAII(problem)
algorithm.run(10)
for solution in algorithm.result:
        print(solution.variables)
        print(solution.objectives)
# plot the results using matplotlib
import matplotlib.pyplot as plt

plt.scatter([s.variables for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])

plt.xlabel("$f_1(x)$")
plt.ylabel("$f_2(x)$")
plt.show()

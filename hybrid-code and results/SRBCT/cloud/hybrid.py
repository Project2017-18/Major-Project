from platypus import NSGAII,OMOPSO, Problem, Real,Binary
from platypus.evaluator import Job
from platypus.operators import TournamentSelector, RandomGenerator,\
    DifferentialEvolution, clip, UniformMutation, NonUniformMutation,\
    GAOperator, SBX, PM, UM, PCX, UNDX, SPX, Multimethod
from platypus.core import Algorithm, ParetoDominance, AttributeDominance,\
    AttributeDominance, nondominated_sort, nondominated_prune,\
    nondominated_truncate, nondominated_split, crowding_distance,\
    EPSILON, POSITIVE_INFINITY, Archive, EpsilonDominance, FitnessArchive,\
    Solution, HypervolumeFitnessEvaluator, nondominated_cmp, fitness_key,\
    crowding_distance_key, AdaptiveGridArchive, Selector, EpsilonBoxArchive,\
    PlatypusError
from platypus.types import Real, Binary, Permutation, Subset
from platypus.config import default_variator, default_mutator
import random

class _EvaluateJob(Job):

    def __init__(self, solution):
        super(_EvaluateJob, self).__init__()
        self.solution = solution
        
    def run(self):
        self.solution.evaluate()

class hmoea(object):
    
   
    
    def __init__(self,
                 problem,population_size = 100,nEXA=100,evaluator=None,
                 selector = TournamentSelector(2),
                 variator = None,
                 dominance=None):
        
        self.problem = problem
        self.generator=RandomGenerator()
        self.population_size=population_size
        self.population = [self.generator.generate(self.problem) for _ in range(self.population_size)]
        self.nEXA=nEXA
        self.evaluator=evaluator
        self.selector = selector
        self.variator = variator
        self.dominance=dominance
        if self.dominance is None:
            self.dominance=ParetoDominance()
        if self.evaluator is None:
            from platypus.config import PlatypusConfig
            self.evaluator = PlatypusConfig.default_evaluator
        if self.variator is None:
            self.variator = default_variator(self.problem)

    def evaluate_all(self, solutions):
        unevaluated = [s for s in solutions if not s.evaluated]
        
        jobs = [_EvaluateJob(s) for s in unevaluated]
        results = self.evaluator.evaluate_all(jobs)
            
        # if needed, update the original solution with the results
        for i, result in enumerate(results):
            if unevaluated[i] != result.solution:
                unevaluated[i].variables[:] = result.solution.variables[:]
                unevaluated[i].objectives[:] = result.solution.objectives[:]
                unevaluated[i].constraints[:] = result.solution.constraints[:]
                unevaluated[i].constraint_violation = result.solution.constraint_violation
                unevaluated[i].feasible = result.solution.feasible
                unevaluated[i].evaluated = result.solution.evaluated
        
        
    def run(self,generations=200):
        gen=generations
        EXA=[]
        PBA=[]

        
        #Evaluate population
        self.evaluate_all(self.population)

        #initialize EXA PBA
        PBA=self.population[:]
        nondominated_sort(self.population)
        EXA = nondominated_truncate(self.population, self.nEXA)
        while(gen>0):
            print('gen:'+str(gen))
            # EXA-propagating-mechanism
            offspring = []
        
            while len(offspring) < self.nEXA:
                parents = self.selector.select(self.variator.arity, EXA)
                offspring.extend(self.variator.evolve(parents))
                
            self.evaluate_all(offspring)
            offspring.extend(EXA)

            nondominated_sort(offspring)
            EXA = [x for x in offspring if x.rank==0]
            if(len(EXA)>self.nEXA):
                EXA = nondominated_truncate(offspring, self.nEXA)

            #population-update-mechanism and mutation
            offspring = []
        
            while len(offspring) <= self.population_size:
                parents=[]
                parents.append(self.selector.select_one(EXA))
                parents.append(self.selector.select_one(PBA))
                offspring.extend(self.variator.evolve(parents))

            self.evaluate_all(offspring)
            offspring.extend(PBA)

            #PBA update strategy
            PBA = set(random.sample(offspring, self.population_size)) 
            PBA = [i for i in offspring if i in PBA]

            #EXA update strategy
            dominated_list=[]
            tempEXA=EXA[:]
            for pop in PBA:
                for pop1 in tempEXA:
                    flag1=self.dominance.compare(pop,pop1)
                    flag2=False
                    if (flag1>0):
                        dominated_list=[]
                        flag2=True
                        break;
                    elif(flag1<0):
                        dominated_list.append(pop1)
                if not flag2:
                    EXA.append(pop)
                    EXA = [x for x in EXA if x not in dominated_list]
                    #EXA = [x for x in EXA if x.rank==0]
            if(len(EXA)>self.nEXA):
                nondominated_sort(EXA)            
                EXA = nondominated_truncate(offspring, self.nEXA)
            
            
            gen=gen-1
            self.result=EXA
            

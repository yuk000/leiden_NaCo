'''This file contains a skeleton implementation for the practical assignment 
for the NACO 21/22 course. 

Your Genetic Algorithm should be callable like so:
    >>> problem = ioh.get_problem(...)
    >>> ga = GeneticAlgorithm(...)
    >>> ga(problem)

In order to ensure this, please inherit from the provided Algorithm interface,
in a similar fashion as the RandomSearch Example:
    >>> class GeneticAlgorithm(Algorithm):
    >>>     ...

Please be sure to use don't change this name (GeneticAlgorithm) for your implementation.

If you override the constructor in your algoritm code, please be sure to 
call super().__init__, in order to correctly setup the interface. 

Only use keyword arguments for your __init__ method. This is a requirement for the
test script to properly evaluate your algoritm.
'''

import ioh
import random
from algorithm import Algorithm
from functools import reduce
from time import sleep
import csv
from cellular_automaton import ObjectiveFunction
 
MAX_ITERATIONS = 3000  #1-40000   default 100
INIT_POPULATION_NUMBER = 1000 #1-4000 100
SELECTION_NUMBER = 10     
CROSSOVER_OFFSPRINGS = 200   #number of individuals created by recombination
MUTATION_NUMBER =100      #number of individuals changed through mutation
MUTATION_PROBABILTY = -1 #leave it unchanged to have 1/n
SELECTION_TOURNAMENT_SIZE = 3 #2-inf
N_K_POINTS = 2 #1-10 default 2

class RandomSearch(Algorithm):
    '''An example of Random Search.'''

    def __call__(self, problem: ioh.problem.Integer, arity: int) -> None:
        self.y_best: float = float("inf")
        for iteration in range(MAX_ITERATIONS):
            # Generate a random list of values in the interval [0,arity[
            x: list[int] = [random.randint(0, arity-1) for _ in range(problem.meta_data.n_variables)]
            
            # Call the problem in order to get the y value    
            y: float = problem(x)
            
            # update the current state
            self.y_best = max(self.y_best, y)

class GeneticAlgorithm(Algorithm):
    '''A skeleton (minimal) implementation of your Genetic Algorithm.'''
    def __init__(self) -> None:
        super().__init__( MAX_ITERATIONS if (0<MAX_ITERATIONS<1000000) else 100 ) #calls the parent init and set the n. of interations
        self.init_function=self.random_init
        
        #this is where we select the operators to use in __call__
        self.recombination_function=self.kPointsCrossover_recombination
        self.mutation_function=self.bitFlip_mutation
        self.selection_function=self.rank_selection
        
        #attributes
        self.arity = -1 
        self.init_population_number=(INIT_POPULATION_NUMBER if (0<INIT_POPULATION_NUMBER<4000) else 100)
        self.selection_number=(SELECTION_NUMBER if (0<SELECTION_NUMBER<200) else int(INIT_POPULATION_NUMBER/10) )
        self.population=[] 
        self.offspring_population=[] 
        self.dimensionality=1 
        self.y_best : float = float ("-inf") #ca
        return

    def random_init(self) -> None:
        # Generate a random list of values in the interval [0,arity[
        self.population = [ [random.randint(0,self.arity-1) for _ in range(self.dimensionality)] for _ in range(INIT_POPULATION_NUMBER) ]
        self.offspring_population = self.population
        return
	
    def kPointsCrossover_recombination(self) -> None:
        #number of crossover points
        n_k_points=N_K_POINTS if (0<N_K_POINTS<=10) else 2

        #list of ordered splitting point with [0] and [None] at the extremities of the list
        k_points=random.sample(range(self.dimensionality),n_k_points) 
        k_points.sort()
        k_points = [0] + k_points
        k_points.append(None)
        
        #every recombination produces two children.
        #To keep the number of offsprings consistent between functions we use the half
        for _ in range(int(CROSSOVER_OFFSPRINGS/2)): 
            sampled_parents=random.sample(self.population,2) #select two random individuals
            x_1=[]
            y_1=[]
            
            #append slices of the parents to the childresn. (index%2) and (index+1) flips between 0 and 1 in opposite order
            #k_points contains the splitting indexes
            for index in range(len(k_points)-1):
                x_1 += sampled_parents[index%2][k_points[index]:k_points[index+1]]  
                y_1 += sampled_parents[(index+1)%2][k_points[index]:k_points[index+1]]    

            #add the two newly created individuals to the offspring population
            self.offspring_population.append(x_1)
            self.offspring_population.append(y_1)
        return
	
    def threeParentsCrossover_recombination(self):
    
        for _ in range(CROSSOVER_OFFSPRINGS):
            #take three random individuals
            sampled_parents=random.sample(self.population,3) #
            #apply crossover
            individual = list(map(lambda a, b, c: a if a==b else c,sampled_parents[0],sampled_parents[1],sampled_parents[2]))
            #append to offspring
            self.offspring_population.append(individual)

    def bitFlip_mutation(self) -> None:
        mut_probability = MUTATION_PROBABILTY if MUTATION_PROBABILTY > 0 else 1/self.dimensionality 
        
        #helper function. "flip" every bit in the list with mut_probabilty probability
        def flip_alleles(x: list):
            for i in range(0,self.dimensionality):
                if random.random() < mut_probability:
                    x[i] = (x[i] + 1)%self.arity
            return x

        for _ in range(MUTATION_NUMBER):
                #select random individual
                tmp_index = random.randint(0,len(self.offspring_population)-1) 
                
                #apply helper function to individual
                self.offspring_population[tmp_index]=flip_alleles(self.offspring_population[tmp_index]) #apply function
        return

    def swap_mutation(self) -> list:
        mut_probability = MUTATION_PROBABILTY if MUTATION_PROBABILTY > 0 else 1/self.dimensionality 

        #helper function. swap two random elements in the list given as the argument
        #with mut_probabilty probabilty
        def swap_alleles(x: list):
            a,b=random.choices(range(self.dimensionality),k=2)
            if random.random() < mut_probability:
                x[a],x[b]=x[b],x[a]
            return x
        
        for _ in range(MUTATION_NUMBER):
            #select random individual
            tmp_index = random.randint(0,len(self.offspring_population)-1)
            
            #apply helper function to individual
            self.offspring_population[tmp_index]=swap_alleles(self.offspring_population[tmp_index])
        return
       
    def tournament_selection(self, problem: ioh.problem.Integer) -> None:
        new_population = []
        q_tournament_size = SELECTION_TOURNAMENT_SIZE if (self.dimensionality>SELECTION_TOURNAMENT_SIZE>0) else 2
        
        for _ in range(SELECTION_NUMBER):
            # Get q individuals from whole population
            sampled_whole = random.sample(self.offspring_population,q_tournament_size)
            
            #create array that contains the fitness values for every element in sampled whole
            sampled_fitness = list(map(problem,sampled_whole))

            #take the best out of the q(=tournament_size) and add the corresponding individual in sampled_whole in the new population
            tmp_best=max(sampled_fitness)
            new_population.append(sampled_whole[sampled_fitness.index(tmp_best)])
            
            #update self.y_best(optional)
            self.y_best = max(self.y_best,tmp_best)
        
        #update the populations
        self.population = new_population
        self.offspring_population = []
        return

    def rank_selection(self, problem: ioh.problem.Integer) -> None:        
        fitness_population = []
        new_population = []

        #create array in which each entry contains an individual and its fitting value
        #-> fitness_popoulation entry: [individual,fitness(individual)]
        fitness_population = list( map(lambda i:[problem(i),i],self.offspring_population) )
        fitness_population.sort(key = lambda x : x[0],reverse=True)
        
        #update y_best
        self.y_best = max(self.y_best,fitness_population[0][0])
        
        #add the best individuals 
        for i in range(self.selection_number):
            new_population += [fitness_population[i][1]]
        
        #update populations
        self.population=new_population
        self.offspring_population=[]
        return

    def rouletteWheel_selection(self, problem: ioh.problem.Integer) -> None:
        fitness_population = []
        new_population = []
        
        #-> fitness_popoulation entry: [individual,fitness(individual)]
        fitness_population = list(map(lambda i : [i,problem(i)],self.offspring_population))
        
        #sum of all the fitness values
        sum_of_fitness = reduce(lambda a,b: a+b,map(lambda i: i[1],fitness_population))
        
        #-> [individual,fitness(individual),p], with p = fitness(individual)/sum_of_fitness
        fitness_population = list(map(lambda i : i+[i[1]/sum_of_fitness],fitness_population))

        #filter out the entries with probability p = 0
        #the dummy entry at the start makes it easier to implement later steps
        fitness_population = [[0,0.0,0.0]] + list(filter(lambda i: i[2]>0.0 , fitness_population))

        #->[individual,fitness(individual),z] z = sum of the p of all the previous entries in the list
        for index in range(1,len(fitness_population)):
            fitness_population[index][2]+=fitness_population[index-1][2]     
        
        #iterate selection_number times
        for _ in range(self.selection_number): 
            #draw a number
            rand_float=random.random() * 0.999
            
            #search for the winner of the roulette wheel
            for index in range(1,len(fitness_population)):
                if  fitness_population[index-1][2]< rand_float < fitness_population[index][2]:
                    new_population.append(fitness_population[index][0])
                    break
        
        #update populations
        self.population=new_population
        self.offspring_population=[]
        pass

    def __call__(self, problem: ioh.problem.Integer, arity: int) -> None:
        #get the dimensionality from the problem object
        self.dimensionality=problem.meta_data.n_variables

        self.arity = arity

        self.init_function()  
        for _ in range(self.max_iterations):
            if problem.state.optimum_found == True or problem.state.current_best.y == 1:
                break

            self.selection_function(problem)

            self.recombination_function() 

            self.mutation_function()
        return
    
def main():
    
    #open csv file, every line it's a Dictionary
    try:
        csvfile = open('input_redux.csv', newline='')
        reader = csv.DictReader(csvfile)
    except:
        print("input csv file doesn't exist")
        return
    
    #create ObjectiveFunction object and loggerObject
    OF = ObjectiveFunction()
    logger = ioh.logger.Analyzer(store_positions=True) 
    
    for i,row in enumerate(reader):

        OF.update_variables(row)

        # Instantiate the algorithm
        algorithm = GeneticAlgorithm()

        # Get a problem from the IOHexperimenter environment
        problem = ioh.problem.wrap_integer_problem(
            OF.hamming,     #Select the Objective function
            f"random_hamming{i}",
            60,
            ioh.OptimizationType.Maximization,
            ioh.IntegerConstraint([0]*60, [int(row["k"])-1]*60) 
        )

        problem.attach_logger(logger) 

        # Run the algorithm on the problem
        algorithm(problem,int(row["k"]))

        # Inspect the results
        print("Best solution found:")
        print("".join(map(str, problem.state.current_best.x)))
        print("With an objective value of:", problem.state.current_best.y)
        print("number of evaluation:",problem.state.evaluations)
        problem.reset()


    
if __name__ == '__main__':
    main()

import typing
import ioh
import random
from functools import reduce 
import csv
#from implementation import RandomSearch

#Objective Functions parameters
#OF_arity,OF_rule, OF_t, OF_ct = None, None, None,None 


# all combinations of neighbourhood states for binary and ternary
binary_map = [0]*8		
ternary_map = [0]*27

random.seed(42)

def ter(number):	# decimal to ternary conversion
	if(number == 0):
		return 0
	nums = []
	while (number):
		number, r = divmod(number, 3)
		nums.append(str(r))
	return ''.join(reversed(nums))

# generate map with lookup values for every possible neighbourhood
# based on transition rule
def generate_map(self):
	if (self.arity == 2):
		#convert decimal to binary string of length 8
		number = '{:0>8}'.format(str(bin(self.rule_number))[2:])
		
		#
		for i in range(len(number)):
			binary_map[8-(i+1)] = int(number[i])

	if (self.arity == 3):
		#convert decimal to ternary string of length 27
		number ='{:0>27}'.format(str(ter(self.rule_number)))
		
		#
		for i in range(len(number)):
			ternary_map[27-(i+1)] = int(number[i])

# function to get new value for neighborhood based on the transition rule
def lookup_map(self, neighbour_number):
	index = self.arity ** 3 - int(neighbour_number) - 1
	if (self.arity == 2):
		value = binary_map[index]
	if (self.arity == 3):
		value = ternary_map[index]
	return value


# Object that contains the ObjectiveFunctions and the other methods needed
# the OFs should be calculating the similarity -> higher is better
class ObjectiveFunction:
	#declare variables needed
	def __init__(self):
		self.arity, self.rule, self.t, self.ct = None, None, None, None
	
	def hamming(self,c0_prime: typing.List[int]) -> float:	
		#create the CA and compute the t-nth step given a string c0_prime
		ca_prime = CellularAutomaton(self.arity,self.rule)
		ct_prime = ca_prime(c0_prime,self.t)

		#comparing given CT to CT_prime found using hamming 
		n_of_changes = reduce(lambda x,y:x+y , map(lambda a,b : int(a != b) , ct_prime,self.ct) )
		
		#return real value, relative to the length of the string
		distance = n_of_changes / len(ct_prime)
		return 1-distance 
	
	def r_contiguos(self, c0_prime: typing.List[int]) -> float:
		#create the CA and compute the t-nth step given a string c0_prime
		ca_prime = CellularAutomaton(self.arity,self.rule)
		ct_prime = ca_prime(c0_prime,self.t)
		
		#count the longest commons substring, keeping track of the best found
		counter = 0
		best_found = 0
		for i in range(len(self.ct)):
			if ct_prime[i] == self.ct[i]:
				counter += 1
			else:
				best_found = max(best_found,counter)
				counter = 0
		best_found = max(best_found,counter)
		
		#return real value, relative to the length of the string
		return best_found/len(self.ct)

	#get the variables needed for the Objective functions 
	#from the dictionary passed as argument
	def update_variables(self,row:dict):
		self.arity = int(row['k'])
		self.rule= int(row["rule #"])
		self.t = int(row["T"])

		self.ct = list(map(int , row["CT"][1:-1].split(',')))
		return 

	#Used to double check. print the cellular automaton after T steps
	def debug_test(self,c0: typing.List[int]):
		print()
		print("BEFORE:")
		print("".join(map(str, c0)))
		ca_prime = CellularAutomaton(self.arity,self.rule)
		ct_prime = ca_prime(c0,self.t)
		print("AFTER %i STEPS:" % (self.t))
		print("".join(map(str, ct_prime)))
		print("CT:")
		print("".join(map(str, self.ct)))
		print()

class CellularAutomaton:
	"""1-dimensional Cellular Automaton"""

        # initialize CA with arity and rule number
	def __init__(self, arity: int, rule_number: int):
		self.arity = arity
		self.rule_number = rule_number
		generate_map(self)
	
        # call function to run CA simulation for t timesteps
	def __call__(self, c0: typing.List[int], t: int) -> typing.List[int]:
		self.CA = c0
		self.size = len(c0)
		#print( *self.CA)
		for i in range(t):
			self.update()
			#print( *self.CA)
			
		return self.CA
	
        # simulate single time step
	def update(self):
		nextCA = [0]*self.size
		for i in range(self.size):
			value = 0
			
			# Get neighbourhood value
			
			# Add value of the cell on the left of i
			if (i > 0):
				value += int(self.CA[i-1]) * self.arity * self.arity # 2^2 or 3^2
			
			# Add value of the cell i
			value = int(self.CA[i]) * self.arity	# 2^1 or 3^1
						
			# Add value of the cell on the right of i
			if (i < self.size-1):
				value += int(self.CA[i+1]) # 2^0 or 3^0 (which is 1)
			
			nextCA[i] = lookup_map(self, value)

		self.CA = nextCA

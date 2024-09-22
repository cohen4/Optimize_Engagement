import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import random

def gene_algo(start_pop): # Genetic Algorithm, main function
    generation = x_to_gene(start_pop)
    num_of_stable_gen = 0 # Stop condition, Number of the generations without decrease in standard deviation
    # All the graphs For plotting
    max_graph = []
    avg_graph = []
    avg_plus_graph = []
    avg_minus_graph = []
    j = 1 # Generation index
    max_val, avg_val, avg_val_plus, avg_val_minus, sd_prev, parent_max = find_max_and_avg(generation) # Seeding generation's relevant details
    # sd_prev = standard deviation of the previous generation, In the first iteration the seeding generation will get this var
    max_graph.append(max_val)
    avg_graph.append(avg_val)
    avg_plus_graph.append(avg_val_plus)
    avg_minus_graph.append(avg_val_minus)
    while True:
        if j >= 1500: # Maximum number of iteration
            break
        else: # Create the next generation
            j += 1  # Number of generations
            parent_pro, gene_elite_parent_1, gene_elite_parent_2 = fitness_function_and_elitism(generation)
            # Gets the parents probability to be chosen and 2 elites parents
            next_gen = [] # The offsprings that's continue to the next generation
            while len(next_gen) < 19: # Stop when get to 19 offsprings
                parent_1, parent_2 = sampling_parents(generation, parent_pro) # Chose 2 parents
                cross_point = one_point_crossover() # Gets the crossing point
                offspring_1, offspring_2 = offspring(parent_1, parent_2, cross_point) # Two parents fathers two offsprings
                offspring_1 = mutation(offspring_1) # Changing one bit of the offspring
                offspring_2 = mutation(offspring_2)
                offspring_1_feasibility = offspring_feasibility(offspring_1) # Checking if the offspring meets the  two conditions
                offspring_2_feasibility = offspring_feasibility(offspring_2)
                if offspring_1_feasibility == True:
                    next_gen.append(offspring_1)
                if offspring_2_feasibility == True and len(next_gen) < 19:
                    next_gen.append(offspring_2)
            next_gen = best_offsprings(next_gen) # Getting the 17 best offsprings
            next_gen.append(gene_elite_parent_1) # The highest two parents value continue to the next generation
            next_gen.append(gene_elite_parent_2)
            generation = next_gen # Now the current generation is the elite of parents and offsprings
            max_val, avg_val, avg_val_plus, avg_val_minus, sd_curr, parent_max = find_max_and_avg(generation)
            # sd_curr = Standard deviation of the current generation
            # Gets relevant details for graph and final solution
            max_graph.append(max_val)
            avg_graph.append(avg_val)
            avg_plus_graph.append(avg_val_plus)
            avg_minus_graph.append(avg_val_minus)
        if sd_curr >= sd_prev :
            num_of_stable_gen += 1
            if num_of_stable_gen == 10: # If we have 10 generations in a row break the loop.
                break
        else:
            num_of_stable_gen = 0 # reset the val
        sd_prev = sd_curr
    # Plotting
    number_of_gene = np.arange(1, j+1, 1)
    fig, axs = plt.subplots(1)
    axs.plot(number_of_gene, max_graph, label = "Maximum F")
    axs.plot(number_of_gene, avg_graph, label = "Average F")
    axs.plot(number_of_gene, avg_plus_graph, label="Positive Confidence Interval of the Average", color = "#0072BD")
    axs.plot(number_of_gene, avg_minus_graph, label="Negative Confidence Interval of the Average", color = "#4DBEEE")
    axs.set_ylim(0, 13)
    axs.set_xlim(0, j)
    plt.tight_layout()
    plt.legend()
    plt.show()

    return max_graph, avg_graph, parent_max, max_val


def x_to_gene(answers): # Transform the answers to genes
    answers_to_gene = []
    for answer in answers:
        answer[1] *= 10  # The second variable gets one number after decimal point
        var_1_and_2 = np.zeros(19) # First variable gets the first 11 bits and second variable gets the last 8 bits
        for i in range(11):
            if (answer[0] - 2 ** (10-i)) >= 0: # We set the binary number on base of 2
                var_1_and_2[i] = 1
                answer[0] -= 2**(10-i)
            if answer[0] == 0:
                break
        for k in range(8):
            if (answer[1]- 2**(7-k)) >= 0:
                var_1_and_2[k + 11] = 1
                answer[1] -= 2**(7-k)
            if answer[1] == 0:
                break
        answers_to_gene.append(var_1_and_2)
    return answers_to_gene


def gene_to_x(gene_answer):
    x = 0  # First variable value
    y = 0  # Second variable value
    for i in range(11):
        if gene_answer[i] == 1:
            x += 2 ** (10-i)
    for j in range(8):
        if gene_answer[j+ 11] == 1:
            y += 2 ** (7-j)
    return np.array([x,y/10]) # Dividing variable 2 by ten for getting back to the original answer

def fitness_function_and_elitism(generation):
    total_f = 0 # Sum of parents' values
    parent_pro = [] # Parent probability
    parent_f = []
    for i in range(19):
        parent_value = f(generation[i])
        total_f += parent_value
        parent_f.append(parent_value)
    for k in range(19):
        parent_pro.append(parent_f[k]/total_f)

    sort_parent_f = sorted(parent_f, reverse = True ) # Sorting parents' values from highest to lowest and taking the two highest.
    elite_parent_1 = sort_parent_f[0]
    elite_parent_2 = sort_parent_f[1] # The next generation contains 10% of the best parents. (19*0.1 =~ 2)
    gene_elite_parent_1 = generation[parent_f.index(elite_parent_1)]
    gene_elite_parent_2 = generation[parent_f.index(elite_parent_2)]
    return parent_pro, gene_elite_parent_1, gene_elite_parent_2

def sampling_parents(generation, parent_pro):
    generation_copy = generation.copy() # For not changing the original generation list
    parent_1 = random.choices(generation_copy, weights=parent_pro, k=1)
    generation_array = np.array(generation_copy) # Changing the type to array for removing parent 1. ( it didn't work with list)
    index_to_remove = np.where((generation_array == parent_1[0]).all(axis=1))[0] # Finding the index of parent_1
    generation_array = np.delete(generation_array, index_to_remove, axis=0) # Removing parent 1 from the list because it already chosen
    generation_list = generation_array.tolist() # Return to list type
    parent_2 = random.choices(generation_list, k=1) # All the other parents have equal probability to be chosen
    return parent_1[0], parent_2[0]

def one_point_crossover(): # one point crossover for each of the variables
    cross_point = np.random.randint(1,18)
    # If the point crossover will be in index 0 or 19, the offspring won't be effective because it will swap all the gene
    return cross_point

def offspring(parent1, parent2, cross_point):
    offspring_1 = np.zeros(19)
    offspring_1[0:cross_point] = parent1[0:cross_point] # offspring gets the gene of parent 1 until to the cross point
    offspring_1[cross_point:19] = parent2[cross_point:19] # offspring gets the gene of parent 2 from cross point to the end of the gene

    offspring_2 = np.zeros(19)
    offspring_2[0:cross_point] = parent2[0:cross_point] # Same like parent 1 but reversed
    offspring_2[cross_point:19] = parent1[cross_point:19]
    return offspring_1, offspring_2

def mutation(offspring):
    mutation_point = np.random.randint(0,18)
    if offspring[mutation_point] == 0:
        offspring[mutation_point] = 1
    else:
        offspring[mutation_point] = 0
    return offspring

def f(parent_gene): # The target function
    # Parameters from part A of the project
    a = 100
    b = 5
    c = 70
    TF = 10000
    parent = gene_to_x(parent_gene) # returns the x and y values
    parent_value = (np.sqrt(a * parent[0] * parent[1]) + np.sqrt(b * parent[0]) + np.sqrt(c * parent[1])) * 100 / TF
    return parent_value

def offspring_feasibility(offspring):
    check_sol = gene_to_x(offspring)
    # Setting the relevant parameters of constrains
    MT = 16
    OS = 200
    Budget = 2000

    if check_sol[1]+(check_sol[0]/OS) <= MT: # Constrain 1
        if check_sol[0] <= Budget: # Constrain 2
            if check_sol[0] >= 0 and check_sol[1] >= 0: # NonNegative Constrains
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def best_offsprings(offsprings):
    offspring_f = []
    next_offsprings = [] # The offsprings that's move to the next generation
    for i in range(19):
        offspring_value = f(offsprings[i]) # Gets the value of the offspring
        offspring_f.append(offspring_value) # Append to the offspring's list
    sort_offspring_f = sorted(offspring_f, reverse = True) # Sorted from highest  value offspring to the lowest
    for i in range(17):
        max_i_offspring = sort_offspring_f[i] # The highest value offspring in index i
        index_max_i = offspring_f.index(max_i_offspring) # Returns the index of the offspring in the original list
        next_offsprings.append(offsprings[index_max_i]) # Append the gene of the relevant offspring
    return next_offsprings

def find_max_and_avg(generation): # Finding the maximum and average values
    parent_f = [] # In this function we relate to the all generation as parents
    total_f = 0
    for i in range(19):
        parent_val = f(generation[i]) # Gets parent value
        parent_f.append(parent_val)
        total_f += parent_val # For calculating the average
    max_val = max(parent_f) # Get the highest value in the generation
    avg_val = total_f / 19
    a = 1.0 * np.array(parent_f)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1+ 0.95) / 2, n-1) # Calculating the confidence interval
    index_parent_max = parent_f.index(max_val) # Get the highest parent index
    parent_max = gene_to_x(generation[index_parent_max]) # Retrieve the highest value point(x,y) in the current generation
    return max_val, avg_val, avg_val + h, avg_val - h , h, parent_max


# We have 19 bits in our genetic code so the seed population contains 19 initial solutions
start_pop = [np.array([1999,5.9]), np.array([54,5]), np.array([434,3]), np.array([1727,4.2]), np.array([656,12.1]), np.array([764,4.6]),
                np.array([922,8.2]), np.array([1282,8.3]), np.array([1692,2.6]), np.array([111,11.3]), np.array([1021, 4.1]),
                    np.array([1081,4.9]), np.array([639,0.8]), np.array([222,10]), np.array([755,9.7]), np.array([917, 9.9]),
                        np.array([1267, 3.7]), np.array([961, 2.4]), np.array([556, 6.1])]

max_graph, avg_graph, parent_max, max_val = gene_algo(start_pop)

print("Optimal_Point: ", parent_max)
print("Optimal Value: ", np.round(max_val,3))
print("Max F Per Generation: ", np.round(max_graph,3))
print("Average F Per Generation: ", np.round(avg_graph,3))










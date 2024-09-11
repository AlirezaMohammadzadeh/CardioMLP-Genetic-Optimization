from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Create the Fitness and Individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate(individual):
    # Ensure the number of hidden layers is an integer
    num_hidden_layers = int(individual[-4])
    
    # Extract the sizes of the hidden layers based on the number of hidden layers
    hidden_layers = tuple(individual[:num_hidden_layers])
    
    # Extract other hyperparameters
    activation = ['identity', 'logistic', 'tanh', 'relu'][individual[-3]]
    solver = ['lbfgs', 'sgd', 'adam'][individual[-2]]
    alpha = individual[-1]

    # Initialize the MLPClassifier with the hyperparameters
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation,
                          solver=solver, alpha=alpha, max_iter=500, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred = model.predict(X_val)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    
    # Return the fitness score (accuracy)
    return accuracy,

# Define the genetic algorithm components
toolbox = base.Toolbox()

# Define individual components: up to 6 hidden layers, activation, solver, and alpha
toolbox.register("attr_hidden_layer_size", random.randint, 10, 200)
toolbox.register("attr_num_hidden_layers", random.randint, 1, 6)  # 1 to 6 hidden layers
toolbox.register("attr_activation", random.randint, 0, 3)
toolbox.register("attr_solver", random.randint, 0, 2)
toolbox.register("attr_alpha", random.uniform, 0.0001, 0.01)

# Define the individual and population
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_hidden_layer_size,
                  toolbox.attr_hidden_layer_size,
                  toolbox.attr_hidden_layer_size,
                  toolbox.attr_hidden_layer_size,
                  toolbox.attr_hidden_layer_size,
                  toolbox.attr_hidden_layer_size,
                  toolbox.attr_num_hidden_layers,
                  toolbox.attr_activation,
                  toolbox.attr_solver,
                  toolbox.attr_alpha), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Custom mutation function to handle the number of hidden layers
def mutate_individual(individual):
    # Ensure the number of hidden layers is used correctly
    num_layers = int(individual[-4])
    for i in range(num_layers):
        individual[i] = random.randint(10, 200)
    for i in range(num_layers, 6):
        individual[i] = 0  # Zero out unused layers


    # Mutate other parameters
    individual[-3] = random.randint(0, 3)
    individual[-2] = random.randint(0, 2)
    individual[-1] = random.uniform(0.0001, 0.01)
    return individual,

# Update mutation function to ensure valid hidden layer sizes
toolbox.register("mutate", tools.mutUniformInt, 
                 low=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0.0001],  # Ensure minimum size of 1 for hidden layers
                 up=[200, 200, 200, 200, 200, 200, 6, 3, 2, 0.01], indpb=0.2)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Main function to run the genetic algorithm
def main():
    random.seed(42)
    
    # Initialize population
    population = toolbox.population(n=20)
    
    # Apply the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
    
    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]
    print("Best individual is:", best_individual)
    print("Best validation accuracy is:", evaluate(best_individual)[0])

if name == "main":
    main()
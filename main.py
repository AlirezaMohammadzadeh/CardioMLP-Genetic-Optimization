import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools, algorithms
import random
import torch

file_path = 'dataset/cardio_train.csv'

df = pd.read_csv(file_path, sep=';')

columns = df.columns.drop('cardio')

df['age'] = df['age'] / 365
df['age'] = df['age'].astype(int)

# group by blood pressure
# 1- Normal (sys <= 120 and dia <= 80)
# 2- at risk (120 < sys <= 140 or 80 < dia <= 90)
# 3- high (sys > 140 or dia > 90)

df['blood_pressure'] = 0

df.loc[(df['ap_hi'] <= 120) & (df['ap_lo'] <= 80), 'blood_pressure'] = 1

df.loc[((df['ap_hi'] > 120) & (df['ap_hi'] <= 140)) | ((df['ap_lo'] > 80) & (df['ap_lo'] <= 90)), 'blood_pressure'] = 2

df.loc[(df['ap_hi'] > 140) | (df['ap_lo'] > 90), 'blood_pressure'] = 3

df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

df.fillna(df.median(), inplace=True)
df = df.drop(columns=['id'],axis=1)

# Extract features (all columns except 'cardio') and target variable ('cardio')
X = df.drop(columns=['cardio'],axis=1)
y = df['cardio']

columns_to_standardize = ['age', 'height','weight','ap_hi','ap_lo','bmi']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])
X_test[columns_to_standardize] = scaler.transform(X_test[columns_to_standardize])
X_val[columns_to_standardize] = scaler.transform(X_val[columns_to_standardize])

# Feature Selection using SelectKBest
print("Performing feature selection...")
selector = SelectKBest(score_func=f_classif, k=8)  # Select top 8 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()].tolist()
print(f"Selected features: {selected_features}")

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Running on CPU instead.")

# Create the Fitness and Individual classes for hyperparameter optimization only
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def custom_mutate(individual, indpb):
    # Mutate the hidden layer sizes and number of layers
    for i in range(5):  # Reduced from 10 to 5 possible hidden layers
        if random.random() < indpb:
            individual[i] = random.randint(10, 100)  # Reduced max size from 200 to 100
    
    if random.random() < indpb:
        individual[5] = random.randint(1, 3)  # Reduced max layers from 10 to 3
    
    if random.random() < indpb:
        individual[6] = random.randint(0, 3)

    if random.random() < indpb:
        individual[7] = random.uniform(0.0001, 0.01)

    return individual,

# Define the neural network model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, activation='relu'):
        super(MLPModel, self).__init__()
        
        # Create list of layers
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'logistic':
                layers.append(nn.Sigmoid())
            prev_size = size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def evaluate(individual):
    hyperparameters = individual
    
    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    
    try:
        # Configure model
        num_hidden_layers = int(individual[5])
        hidden_layers = tuple(hyperparameters[:num_hidden_layers])
        activation = ['identity', 'logistic', 'tanh', 'relu'][individual[6]]
        alpha = individual[7]
        
        # Create and move model to GPU
        model = MLPModel(X_train_selected.shape[1], hidden_layers, activation).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)
        
        # Training loop - reduced epochs from 100 to 20
        model.train()
        
        for epoch in range(20):
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Calculate final training accuracy
        with torch.no_grad():
            outputs = model(X_train_tensor)
            predictions = (outputs > 0.5).float().squeeze()
            accuracy = (predictions == y_train_tensor).float().mean().item()
            
        return accuracy,
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0.0,

# Define the genetic algorithm components
toolbox = base.Toolbox()

# Define individual components - only hyperparameters now
toolbox.register("attr_hidden_layer_size", random.randint, 10, 100)
toolbox.register("attr_num_hidden_layers", random.randint, 1, 3)
toolbox.register("attr_activation", random.randint, 0, 3)
toolbox.register("attr_alpha", random.uniform, 0.0001, 0.01)

# Create hyperparameter attributes
hidden_layer_attrs = [toolbox.attr_hidden_layer_size for _ in range(5)]
other_attrs = [toolbox.attr_num_hidden_layers, toolbox.attr_activation, toolbox.attr_alpha]
all_attrs = hidden_layer_attrs + other_attrs

toolbox.register("individual", tools.initCycle, creator.Individual, all_attrs, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def log_stats(gen, population, fits):
    """Log statistics for each generation"""
    fit_mins = min(fits)
    fit_maxs = max(fits)
    fit_means = sum(fits) / len(population)
    fit_std = np.std(fits)
    
    print(f"Gen {gen}: ")
    print(f"  Min: {fit_mins:.4f}")
    print(f"  Max: {fit_maxs:.4f}")
    print(f"  Avg: {fit_means:.4f}")
    print(f"  Std: {fit_std:.4f}")
    print("------------------------")

def train_best_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, criterion, optimizer, num_epochs=50):  # Reduced from 1000 to 50
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor.unsqueeze(1))
        train_loss.backward()
        optimizer.step()
        
        # Calculate training metrics
        train_preds = (train_outputs > 0.5).float().squeeze()
        train_acc = (train_preds == y_train_tensor).float().mean().item()
        train_losses.append(train_loss.item())
        train_accuracies.append(train_acc)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor.unsqueeze(1))
            val_preds = (val_outputs > 0.5).float().squeeze()
            val_acc = (val_preds == y_val_tensor).float().mean().item()
            val_losses.append(val_loss.item())
            val_accuracies.append(val_acc)
        
        if epoch % 10 == 0:  # Reduced logging frequency
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def plot_training_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize population - reduced from 10 to 5
    population = toolbox.population(n=5)
    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    # Configure logging
    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max", "avg", "std"

    # Apply the genetic algorithm with logging - reduced to 3 generations
    for gen in range(3):
        # Select the next generation individuals
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population with offspring
        population[:] = offspring
        
        # Gather and print statistics
        fits = [ind.fitness.values[0] for ind in population]
        log_stats(gen, population, fits)
    
    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]
    print("\nBest individual is:", best_individual)
    
    # Convert to CUDA tensors
    X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_selected, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)
    
    # Configure best model
    num_hidden_layers = int(best_individual[5])
    hidden_layers = tuple(best_individual[:num_hidden_layers])
    activation = ['identity', 'logistic', 'tanh', 'relu'][best_individual[6]]
    alpha = best_individual[7]
    
    print(f"\nBest model configuration:")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Activation: {activation}")
    print(f"Alpha (L2 regularization): {alpha}")
    
    # Create and train final model
    best_model = MLPModel(X_train_selected.shape[1], hidden_layers, activation).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=0.001, weight_decay=alpha)
    
    # Train best model with validation
    print("\nTraining best model...")
    train_losses, train_accuracies, val_losses, val_accuracies = train_best_model(
        best_model, X_train_tensor, y_train_tensor, 
        X_val_tensor, y_val_tensor,
        criterion, optimizer
    )

    # Plot training metrics
    plot_training_metrics(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # Evaluate final model
    best_model.eval()
    with torch.no_grad():
        test_outputs = best_model(X_test_tensor)
        test_preds = (test_outputs > 0.5).float().squeeze()
        test_probs = test_outputs.squeeze()
        
        # Move tensors to CPU for metric calculation
        test_preds_cpu = test_preds.cpu().numpy()
        test_probs_cpu = test_probs.cpu().numpy()
        y_test_cpu = y_test_tensor.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_cpu, test_preds_cpu)
        precision = precision_score(y_test_cpu, test_preds_cpu)
        recall = recall_score(y_test_cpu, test_preds_cpu)
        f1 = f1_score(y_test_cpu, test_preds_cpu)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Print selected features
    print(f"\nSelected Features ({len(selected_features)}): {selected_features}")

if __name__ == "__main__":
    main()





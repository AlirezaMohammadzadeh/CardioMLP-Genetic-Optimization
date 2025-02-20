import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pygad
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
X_test[columns_to_standardize] = scaler.fit_transform(X_test[columns_to_standardize])
X_val[columns_to_standardize] = scaler.fit_transform(X_val[columns_to_standardize])


# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Running on CPU instead.")


# Create the Fitness and Individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def custom_mutate(individual, indpb):
    # Mutate the feature mask (binary values)
    for i in range(num_features):
        if random.random() < indpb:
            individual[i] = 1 - individual[i]  # Flip the binary feature mask
    
    # Mutate the hidden layer sizes and number of layers
    for i in range(num_features, num_features + 10):
        if random.random() < indpb:
            individual[i] = random.randint(10, 200)
    
    if random.random() < indpb:
        individual[num_features + 10] = random.randint(1, 10)
    
    if random.random() < indpb:
        individual[num_features + 11] = random.randint(0, 3)

    if random.random() < indpb:
        individual[num_features + 12] = random.uniform(0.0001, 0.01)

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
    feature_mask = individual[:num_features]
    hyperparameters = individual[num_features:]
    
    # Convert feature mask to boolean array
    feature_mask = np.array(feature_mask, dtype=bool)
    X_train_selected = X_train.iloc[:, feature_mask]
    X_val_selected = X_val.iloc[:, feature_mask]
    
    if X_train_selected.shape[1] == 0:  # If no features selected
        return 0.0,

    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.tensor(X_train_selected.values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_selected.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    try:
        # Configure model
        num_hidden_layers = int(individual[-3])
        hidden_layers = tuple(hyperparameters[:num_hidden_layers])
        activation = ['identity', 'logistic', 'tanh', 'relu'][individual[-2]]
        alpha = individual[-1]  # L2 regularization parameter
        
        # Create and move model to GPU
        model = MLPModel(X_train_selected.shape[1], hidden_layers, activation).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)
        
        # Training loop
        model.train()
        for epoch in range(100):  # Reduced epochs for faster evaluation
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_preds = (val_outputs > 0.5).float().squeeze()
            accuracy = (val_preds == y_val_tensor).float().mean().item()
        
        return accuracy,
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0.0,

# Define the genetic algorithm components
toolbox = base.Toolbox()

# Define individual components
num_features = X_train.shape[1]
toolbox.register("attr_feature", random.randint, 0, 1)
toolbox.register("attr_hidden_layer_size", random.randint, 10, 200)
toolbox.register("attr_num_hidden_layers", random.randint, 1, 10)
toolbox.register("attr_activation", random.randint, 0, 3)
toolbox.register("attr_alpha", random.uniform, 0.0001, 0.01)

# Create feature mask attributes
feature_attrs = [toolbox.attr_feature for _ in range(num_features)]
hidden_layer_attrs = [toolbox.attr_hidden_layer_size for _ in range(10)]
other_attrs = [toolbox.attr_num_hidden_layers, toolbox.attr_activation, toolbox.attr_alpha]
all_attrs = feature_attrs + hidden_layer_attrs + other_attrs

toolbox.register("individual", tools.initCycle, creator.Individual, all_attrs, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize population
    population = toolbox.population(n=30)
    
    # Apply the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=True)
    
    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]
    print("\nBest individual is:", best_individual)
    
    # Get feature mask and evaluate final model
    feature_mask = np.array(best_individual[:num_features], dtype=bool)
    X_train_selected = X_train.iloc[:, feature_mask]
    X_test_selected = X_test.iloc[:, feature_mask]
    
    # Convert to CUDA tensors
    X_train_tensor = torch.tensor(X_train_selected.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_selected.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)
    
    # Configure best model
    num_hidden_layers = int(best_individual[-3])
    hidden_layers = tuple(best_individual[num_features:num_features + num_hidden_layers])
    activation = ['identity', 'logistic', 'tanh', 'relu'][best_individual[-2]]
    alpha = best_individual[-1]
    
    # Create and train final model
    best_model = MLPModel(X_train_selected.shape[1], hidden_layers, activation).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=0.001, weight_decay=alpha)
    
    # Create data loader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train final model
    best_model.train()
    for epoch in range(1000):  # More epochs for final model
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = best_model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
    
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
        auc = roc_auc_score(y_test_cpu, test_probs_cpu)
        brier = brier_score_loss(y_test_cpu, test_probs_cpu)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    
    # Print selected features
    selected_features = X_train.columns[feature_mask].tolist()
    print("\nSelected Features:", selected_features)

if __name__ == "__main__":
    main()





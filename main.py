import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools, algorithms
import random
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

file_path = 'dataset/cardio_train.csv'

df = pd.read_csv(file_path, sep=';')

print("Dataset shape:", df.shape)
print("Target distribution:")
print(df['cardio'].value_counts(normalize=True))

# Remove outliers using IQR method
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Remove extreme outliers for blood pressure and weight
outlier_columns = ['ap_hi', 'ap_lo', 'weight', 'height']
print(f"Before outlier removal: {len(df)} samples")
df = remove_outliers(df, outlier_columns)
print(f"After outlier removal: {len(df)} samples")

# Convert age from days to years
df['age'] = df['age'] / 365
df['age'] = df['age'].astype(int)

# Enhanced feature engineering
# Blood pressure categories
df['blood_pressure'] = 0
df.loc[(df['ap_hi'] <= 120) & (df['ap_lo'] <= 80), 'blood_pressure'] = 1
df.loc[((df['ap_hi'] > 120) & (df['ap_hi'] <= 140)) | ((df['ap_lo'] > 80) & (df['ap_lo'] <= 90)), 'blood_pressure'] = 2
df.loc[(df['ap_hi'] > 140) | (df['ap_lo'] > 90), 'blood_pressure'] = 3

# BMI and BMI categories
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
df['bmi_category'] = 0
df.loc[df['bmi'] < 18.5, 'bmi_category'] = 1  # Underweight
df.loc[(df['bmi'] >= 18.5) & (df['bmi'] < 25), 'bmi_category'] = 2  # Normal
df.loc[(df['bmi'] >= 25) & (df['bmi'] < 30), 'bmi_category'] = 3  # Overweight
df.loc[df['bmi'] >= 30, 'bmi_category'] = 4  # Obese

# Pulse pressure (important cardiovascular indicator)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

# Age groups
df['age_group'] = 0
df.loc[df['age'] < 45, 'age_group'] = 1
df.loc[(df['age'] >= 45) & (df['age'] < 55), 'age_group'] = 2
df.loc[(df['age'] >= 55) & (df['age'] < 65), 'age_group'] = 3
df.loc[df['age'] >= 65, 'age_group'] = 4

# Risk factors combination
df['risk_factors'] = df['smoke'] + df['alco'] + (df['cholesterol'] > 1).astype(int) + (df['gluc'] > 1).astype(int)

# Interaction features
df['age_bmi'] = df['age'] * df['bmi']
df['bp_age'] = df['blood_pressure'] * df['age']

df.fillna(df.median(), inplace=True)
df = df.drop(columns=['id'], axis=1)

# Extract features and target
X = df.drop(columns=['cardio'], axis=1)
y = df['cardio']

print(f"Final feature set: {list(X.columns)}")
print(f"Number of features: {X.shape[1]}")

# Use more robust scaling
columns_to_standardize = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure', 'age_bmi', 'bp_age']

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

# Use RobustScaler for better outlier handling
scaler = RobustScaler()
X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])
X_test[columns_to_standardize] = scaler.transform(X_test[columns_to_standardize])
X_val[columns_to_standardize] = scaler.transform(X_val[columns_to_standardize])

# Feature Selection using multiple methods
print("Performing feature selection...")

# Method 1: SelectKBest
selector_kbest = SelectKBest(score_func=f_classif, k=12)  # Select more features
X_train_kbest = selector_kbest.fit_transform(X_train, y_train)
selected_features_kbest = X_train.columns[selector_kbest.get_support()].tolist()

# Method 2: Random Forest feature importance
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

# Select top features from RF
top_rf_features = feature_importance.head(12)['feature'].tolist()

# Combine features from both methods
combined_features = list(set(selected_features_kbest + top_rf_features))
print(f"Selected features: {combined_features}")

# Use combined features
X_train_selected = X_train[combined_features].values
X_val_selected = X_val[combined_features].values
X_test_selected = X_test[combined_features].values

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enhanced neural network architecture
class ImprovedMLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, activation='relu', dropout_rate=0.3):
        super(ImprovedMLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))  # Add batch normalization
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'logistic':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01))
            
            layers.append(nn.Dropout(dropout_rate))  # Add dropout
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Create the Fitness and Individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def custom_mutate(individual, indpb):
    # Mutate the hidden layer sizes
    for i in range(5):
        if random.random() < indpb:
            individual[i] = random.randint(20, 150)  # Increased range
    
    if random.random() < indpb:
        individual[5] = random.randint(2, 4)  # 2-4 layers
    
    if random.random() < indpb:
        individual[6] = random.randint(0, 4)  # Added leaky_relu
    
    if random.random() < indpb:
        individual[7] = random.uniform(0.0001, 0.01)  # L2 regularization
    
    if random.random() < indpb:
        individual[8] = random.uniform(0.1, 0.5)  # Dropout rate

    return individual,

def evaluate(individual):
    try:
        # Configure model
        num_hidden_layers = int(individual[5])
        hidden_layers = tuple(individual[:num_hidden_layers])
        activation = ['relu', 'logistic', 'tanh', 'leaky_relu'][individual[6] % 4]  # Fix index error
        alpha = individual[7]
        dropout_rate = individual[8]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_selected, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)
        
        # Create model
        model = ImprovedMLPModel(X_train_selected.shape[1], hidden_layers, activation, dropout_rate).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Training with early stopping
        best_val_acc = 0
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(50):  # Increased epochs
            # Training
            model.train()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_preds = (val_outputs > 0.5).float().squeeze()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
        
        return best_val_acc,
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0.0,

# Define the genetic algorithm components
toolbox = base.Toolbox()

# Enhanced hyperparameter ranges
toolbox.register("attr_hidden_layer_size", random.randint, 20, 150)
toolbox.register("attr_num_hidden_layers", random.randint, 2, 4)
toolbox.register("attr_activation", random.randint, 0, 4)
toolbox.register("attr_alpha", random.uniform, 0.0001, 0.01)
toolbox.register("attr_dropout", random.uniform, 0.1, 0.5)

# Create hyperparameter attributes
hidden_layer_attrs = [toolbox.attr_hidden_layer_size for _ in range(5)]
other_attrs = [toolbox.attr_num_hidden_layers, toolbox.attr_activation, toolbox.attr_alpha, toolbox.attr_dropout]
all_attrs = hidden_layer_attrs + other_attrs

toolbox.register("individual", tools.initCycle, creator.Individual, all_attrs, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, indpb=0.3)
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

def train_best_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, criterion, optimizer, num_epochs=100):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15
    
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
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 20 == 0:
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
    
    # Initialize population - increased for better exploration
    population = toolbox.population(n=8)
    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    # Configure logging
    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max", "avg", "std"

    # Apply the genetic algorithm - increased generations
    print("Starting genetic algorithm optimization...")
    for gen in range(5):
        # Select the next generation individuals
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.3)
        
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
    activation = ['relu', 'logistic', 'tanh', 'leaky_relu'][best_individual[6] % 4]  # Fix index error
    alpha = best_individual[7]
    dropout_rate = best_individual[8]
    
    print(f"\nBest model configuration:")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Activation: {activation}")
    print(f"Alpha (L2 regularization): {alpha}")
    print(f"Dropout rate: {dropout_rate}")
    
    # Create and train final model
    best_model = ImprovedMLPModel(X_train_selected.shape[1], hidden_layers, activation, dropout_rate).to(device)
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
    
    print("\nFinal Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Print feature information
    print(f"\nSelected Features ({len(combined_features)}): {combined_features}")
    print(f"Feature importance (top 10):")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    main()





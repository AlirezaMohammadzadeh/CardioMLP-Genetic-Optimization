import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from deap import base, creator, tools, algorithms
import random
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

file_path = 'dataset/cardio_train.csv'

df = pd.read_csv(file_path, sep=';')

print("Dataset shape:", df.shape)
print("Target distribution:")
print(df['cardio'].value_counts(normalize=True))

# Enhanced outlier removal
def remove_outliers_advanced(df, columns, method='iqr'):
    df_clean = df.copy()
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'zscore':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Remove extreme outliers
outlier_columns = ['ap_hi', 'ap_lo', 'weight', 'height']
print(f"Before outlier removal: {len(df)} samples")
df = remove_outliers_advanced(df, outlier_columns, method='iqr')
print(f"After outlier removal: {len(df)} samples")

# Convert age from days to years
df['age'] = df['age'] / 365
df['age'] = df['age'].astype(int)

# Advanced feature engineering
print("Creating advanced features...")

# Blood pressure categories (more granular)
df['bp_systolic_cat'] = pd.cut(df['ap_hi'], bins=[0, 120, 130, 140, 160, 180, 999], 
                               labels=[1, 2, 3, 4, 5, 6])
df['bp_diastolic_cat'] = pd.cut(df['ap_lo'], bins=[0, 80, 85, 90, 100, 110, 999], 
                                labels=[1, 2, 3, 4, 5, 6])

# BMI categories (WHO standard)
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 999], 
                           labels=[1, 2, 3, 4, 5])

# Pulse pressure and mean arterial pressure
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['mean_arterial_pressure'] = df['ap_lo'] + (df['pulse_pressure'] / 3)

# Age groups (more specific)
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 999], 
                        labels=[1, 2, 3, 4, 5])

# Body Surface Area (Mosteller formula)
df['bsa'] = np.sqrt((df['height'] * df['weight']) / 3600)

# Cardiovascular risk factors
df['risk_factors'] = (df['smoke'] + df['alco'] + 
                     (df['cholesterol'] > 1).astype(int) + 
                     (df['gluc'] > 1).astype(int))

# Advanced interaction features
df['age_bmi_interaction'] = df['age'] * df['bmi']
df['bp_bmi_interaction'] = df['pulse_pressure'] * df['bmi']
df['age_bp_interaction'] = df['age'] * df['pulse_pressure']
df['gender_age_interaction'] = df['gender'] * df['age']

# Health score (composite feature)
df['health_score'] = (df['active'] * 2 - df['smoke'] - df['alco'] - 
                     (df['cholesterol'] > 1).astype(int) - 
                     (df['gluc'] > 1).astype(int))

# Polynomial features for important variables
df['age_squared'] = df['age'] ** 2
df['bmi_squared'] = df['bmi'] ** 2
df['bp_squared'] = df['pulse_pressure'] ** 2

# Convert categorical to numeric
df['bp_systolic_cat'] = df['bp_systolic_cat'].astype(int)
df['bp_diastolic_cat'] = df['bp_diastolic_cat'].astype(int)
df['bmi_category'] = df['bmi_category'].astype(int)
df['age_group'] = df['age_group'].astype(int)

df.fillna(df.median(), inplace=True)
df = df.drop(columns=['id'], axis=1)

# Extract features and target
X = df.drop(columns=['cardio'], axis=1)
y = df['cardio']

print(f"Final feature set: {list(X.columns)}")
print(f"Number of features: {X.shape[1]}")

# Advanced scaling with multiple methods
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 
                     'pulse_pressure', 'mean_arterial_pressure', 'bsa',
                     'age_bmi_interaction', 'bp_bmi_interaction', 'age_bp_interaction',
                     'age_squared', 'bmi_squared', 'bp_squared']

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

# Use QuantileTransformer for better normalization
scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_val_scaled = X_val.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
X_val_scaled[numerical_features] = scaler.transform(X_val[numerical_features])

# Advanced feature selection ensemble
print("Performing advanced feature selection...")

# Method 1: SelectKBest
selector_kbest = SelectKBest(score_func=f_classif, k=15)
X_train_kbest = selector_kbest.fit_transform(X_train_scaled, y_train)
features_kbest = X_train_scaled.columns[selector_kbest.get_support()].tolist()

# Method 2: Random Forest importance
rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_scaled, y_train)
feature_importance = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

# Method 3: Gradient Boosting importance
gb_selector = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_selector.fit(X_train_scaled, y_train)
gb_importance = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'importance': gb_selector.feature_importances_
}).sort_values('importance', ascending=False)

# Method 4: L1 regularization feature selection
lasso_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
lasso_selector.fit(X_train_scaled, y_train)
features_lasso = X_train_scaled.columns[lasso_selector.get_support()].tolist()

# Combine all methods
top_rf_features = feature_importance.head(15)['feature'].tolist()
top_gb_features = gb_importance.head(15)['feature'].tolist()

# Weighted feature selection
all_features = set(features_kbest + top_rf_features + top_gb_features + features_lasso)
feature_counts = {}
for feature in all_features:
    count = 0
    if feature in features_kbest: count += 1
    if feature in top_rf_features: count += 1
    if feature in top_gb_features: count += 1
    if feature in features_lasso: count += 1
    feature_counts[feature] = count

# Select features that appear in at least 2 methods
selected_features = [f for f, count in feature_counts.items() if count >= 2]
print(f"Selected features ({len(selected_features)}): {selected_features}")

# Use selected features
X_train_selected = X_train_scaled[selected_features].values
X_val_selected = X_val_scaled[selected_features].values
X_test_selected = X_test_scaled[selected_features].values

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Advanced neural network with residual connections
class AdvancedMLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, activation='relu', dropout_rate=0.3):
        super(AdvancedMLPModel, self).__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_size = input_size
        
        for i, size in enumerate(hidden_layers):
            self.layers.append(nn.Linear(prev_size, size))
            self.batch_norms.append(nn.BatchNorm1d(size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        self.output_layer = nn.Linear(prev_size, 1)
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'logistic':
            return nn.Sigmoid()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.01)
        elif activation == 'swish':
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def forward(self, x):
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            residual = x if i > 0 and x.size(1) == layer.out_features else None
            
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
            x = dropout(x)
            
            # Add residual connection if possible
            if residual is not None and residual.size(1) == x.size(1):
                x = x + residual
        
        x = self.output_layer(x)
        return torch.sigmoid(x)

# Create the Fitness and Individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def custom_mutate(individual, indpb):
    # Mutate the hidden layer sizes
    for i in range(6):  # Increased layers
        if random.random() < indpb:
            individual[i] = random.randint(32, 256)  # Wider range
    
    if random.random() < indpb:
        individual[6] = random.randint(2, 5)  # 2-5 layers
    
    if random.random() < indpb:
        individual[7] = random.randint(0, 5)  # Added swish activation
    
    if random.random() < indpb:
        individual[8] = random.uniform(0.0001, 0.01)  # L2 regularization
    
    if random.random() < indpb:
        individual[9] = random.uniform(0.1, 0.5)  # Dropout rate
    
    if random.random() < indpb:
        individual[10] = random.uniform(0.0001, 0.01)  # Learning rate

    return individual,

def evaluate(individual):
    try:
        # Configure model
        num_hidden_layers = int(individual[6])
        hidden_layers = tuple(individual[:num_hidden_layers])
        activation = ['relu', 'logistic', 'tanh', 'leaky_relu', 'swish'][individual[7] % 5]
        alpha = individual[8]
        dropout_rate = individual[9]
        learning_rate = individual[10]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_selected, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)
        
        # Create model
        model = AdvancedMLPModel(X_train_selected.shape[1], hidden_layers, activation, dropout_rate).to(device)
        
        # Weighted loss for class imbalance
        class_weights = torch.tensor([1.0, 1.0]).to(device)  # Adjust if needed
        criterion = nn.BCELoss()
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=alpha)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=learning_rate*0.1)
        
        # Training with early stopping
        best_val_acc = 0
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(75):  # Increased epochs
            # Training
            model.train()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_preds = (val_outputs.squeeze() > 0.5).float()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
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
toolbox.register("attr_hidden_layer_size", random.randint, 32, 256)
toolbox.register("attr_num_hidden_layers", random.randint, 2, 5)
toolbox.register("attr_activation", random.randint, 0, 5)
toolbox.register("attr_alpha", random.uniform, 0.0001, 0.01)
toolbox.register("attr_dropout", random.uniform, 0.1, 0.5)
toolbox.register("attr_lr", random.uniform, 0.0001, 0.01)

# Create hyperparameter attributes
hidden_layer_attrs = [toolbox.attr_hidden_layer_size for _ in range(6)]
other_attrs = [toolbox.attr_num_hidden_layers, toolbox.attr_activation, 
               toolbox.attr_alpha, toolbox.attr_dropout, toolbox.attr_lr]
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

def train_ensemble_model(best_individual):
    """Train ensemble of models for better performance"""
    
    # Configure best model
    num_hidden_layers = int(best_individual[6])
    hidden_layers = tuple(best_individual[:num_hidden_layers])
    activation = ['relu', 'logistic', 'tanh', 'leaky_relu', 'swish'][best_individual[7] % 5]
    alpha = best_individual[8]
    dropout_rate = best_individual[9]
    learning_rate = best_individual[10]
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_selected, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)
    
    # Train multiple models with different seeds
    models = []
    seeds = [42, 123, 456, 789, 999]
    
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        model = AdvancedMLPModel(X_train_selected.shape[1], hidden_layers, activation, dropout_rate).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=alpha)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=learning_rate*0.1)
        
        # Training
        best_val_acc = 0
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(150):  # More epochs for final training
            model.train()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_preds = (val_outputs.squeeze() > 0.5).float()
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
        
        models.append(model)
        print(f"Model {len(models)} trained with validation accuracy: {best_val_acc:.4f}")
    
    # Ensemble prediction
    ensemble_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor).squeeze()
            ensemble_preds.append(preds.cpu().numpy())
    
    # Average predictions
    final_preds = np.mean(ensemble_preds, axis=0)
    final_preds_binary = (final_preds > 0.5).astype(int)
    
    return final_preds_binary, final_preds

def main():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize population
    population = toolbox.population(n=10)  # Increased population
    
    # Apply the genetic algorithm
    print("Starting genetic algorithm optimization...")
    for gen in range(7):  # More generations
        # Select the next generation individuals
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.6, mutpb=0.4)
        
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
    
    # Train ensemble model
    print("\nTraining ensemble model...")
    final_preds_binary, final_preds_proba = train_ensemble_model(best_individual)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, final_preds_binary)
    precision = precision_score(y_test, final_preds_binary)
    recall = recall_score(y_test, final_preds_binary)
    f1 = f1_score(y_test, final_preds_binary)
    auc = roc_auc_score(y_test, final_preds_proba)
    
    print("\nFinal Ensemble Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Print feature information
    print(f"\nSelected Features ({len(selected_features)}): {selected_features}")
    print(f"\nTop 10 most important features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    main()





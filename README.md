# MLP with Genetic Algorithm-Based Hyperparameter Tuning for Cardiovascular Disease Prediction

## Overview

This project focuses on optimizing a Multilayer Perceptron (MLP) model for predicting cardiovascular diseases (CVD) using genetic algorithm-based hyperparameter tuning. The dataset has been preprocessed, analyzed, and visualized to enhance model performance. The optimized MLP outperforms the best XGBoost model on the same dataset.

## Dataset

- **Dataset**: Cardiovascular Disease (CVD) dataset
- **Preprocessing**: The dataset was preprocessed and cleaned before model training. Various preprocessing steps included feature scaling, handling missing values, and encoding categorical variables.

## Hyperparameter Optimization

Hyperparameters tuned using a genetic algorithm include:

- **Number of Hidden Layers**: Optimized to 10 layers
- **Neurons per Layer**: `[35, 7, 157, 176, 26, 151, 151, 12, 83, 30]`
- **Activation Function**: ReLU
- **Learning Rate**: `0.007709127453308716`

The genetic algorithm parameters used were:
- **Number of Generations**: 50
- **Population Size**: 30

## Results

- **Best Validation Accuracy**: `0.742`
- **Comparison**: The optimized MLP model achieved a validation accuracy that is 0.12 higher than the best result obtained with XGBoost on the same dataset.

## Jupyter Notebook

The project is implemented in a Jupyter Notebook, which includes the following sections:

1. **Data Preprocessing**: Code for cleaning and preparing the CVD dataset.
2. **Feature Selection**: Techniques applied before hyperparameter tuning.
3. **Genetic Algorithm for Hyperparameter Tuning**: Implementation and configuration of the genetic algorithm.
4. **MLP Model Training**: Configuration and training of the MLP model with optimized hyperparameters.
5. **Evaluation**: Performance evaluation and results generation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


# Collaborative Filtering Recommender System

This repository contains a Python implementation of a collaborative filtering recommender system. The system is designed to predict user-item ratings using both user-based and item-based approaches, and then combine them with a hybrid method. It also provides functionality for evaluating the model using RMSE (Root Mean Square Error).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Code Overview](#code-overview)
- [License](#license)

---

## Overview

The core of this project is the `CollaborativeFiltering` class, which:
- Computes the global average rating, as well as user and item biases.
- Uses adjusted cosine similarity to compute similarities between users and items.
- Makes predictions based on user-based, item-based, or a hybrid approach.
- Evaluates the prediction accuracy using RMSE.

A main script ties everything together: it loads training and test data from CSV files, predicts ratings for the test set, computes the RMSE, and writes the results to an output CSV file.

---

## Features

- **Bias Computation:** Automatically computes the global average rating, user biases (difference between a user's average and the global average), and item biases.
- **Similarity Measures:** Uses adjusted cosine similarity to calculate:
  - **User Similarity:** Adjusts for item bias.
  - **Item Similarity:** Adjusts for user bias.
- **Prediction Methods:** 
  - **User-based Prediction:** Predicts ratings using similar users.
  - **Item-based Prediction:** Predicts ratings using similar items.
  - **Hybrid Prediction:** Combines both methods using configurable weights.
- **Performance Evaluation:** Calculates the RMSE between predicted and actual ratings.
- **Data Loading:** Provides utility functions to load training and test datasets from CSV files.
- **Progress Visualization:** Uses the `tqdm` library to show progress when predicting ratings.

---

## Prerequisites

- **Python 3.x**
- **Libraries:**  
  - `math`
  - `csv`
  - `os`
  - `tqdm`

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install the required Python library:**

   ```bash
   pip install tqdm
   ```

---

## Data Format

### Training Data:
CSV file with each row formatted as:

```
user_id, item_id, rating
```

The file can have a header row. If present, the script will automatically detect and skip it.

### Test Data:
CSV file with each row formatted as:

```
user_id, item_id, actual_rating
```

Similar to the training data, a header row is supported.

Place your training and test CSV files in a folder named `data` within the project directory.

---

## Usage

To run the recommender system, execute the script:

```bash
python script_name.py
```

Replace `script_name.py` with the name of your Python file (e.g., `main.py`).

The script will:
- Load the training and test data.
- Initialize the collaborative filtering model and compute biases.
- Predict ratings using the hybrid method.
- Compute and display the total RMSE.
- Save the predictions along with the actual ratings and RMSE in a CSV file at `data/predictions.csv`.

---

## Hyperparameters

### Number of Neighbors (`k`):
- Default is set to 15 for both user-based and item-based predictions.

### Hybrid Weights:
- The hybrid prediction combines user-based and item-based predictions using weights calculated based on hypothetical RMSE values (e.g., 0.4 for user-based and 0.6 for item-based).
- Ensure that the weights sum to 1. Adjust these values as needed for your dataset.

---

## Code Overview

### CollaborativeFiltering Class:

- **Initialization:** Computes the global average, user biases, and item biases from the training data.

- **Similarity Computation:**
  - `compute_user_similarity`: Adjusts ratings by subtracting the item's average rating.
  - `compute_item_similarity`: Adjusts ratings by subtracting the user's average rating.

- **Prediction Methods:**
  - `predict_user_based_rating`
  - `predict_item_based_rating`
  - `compute_hybrid_prediction`

- **Evaluation:**
  - `calculate_rmse`: Computes RMSE between the actual and predicted ratings.

- **Data Loading Utilities:**
  - `load_training_data`: Reads training data from CSV.
  - `load_test_data`: Reads test data from CSV.

### Main Workflow:

- Loads the training and test data.
- Computes predictions for each user-item pair in the test set.
- Calculates the overall RMSE.
- Saves the predictions and RMSE to an output CSV file.

---

## License

This project is licensed under the MIT License.

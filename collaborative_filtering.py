import math
import csv
import os
from tqdm import tqdm

class CollaborativeFiltering:
    def __init__(self, user_item_ratings):
        """
        Initialize with training data.
        Expected format: {user_id: {item_id: rating, ...}, ...}
        Computes the global average rating, user bias, and item bias for baseline predictors.
        """
        self.user_item_ratings = user_item_ratings
        
        # Compute global average rating
        total, count = 0.0, 0
        for ratings in user_item_ratings.values():
            for rating in ratings.values():
                total += rating
                count += 1
        self.global_avg = total / count if count > 0 else 0.0
        
        # Compute user bias: user_avg - global_avg
        self.user_bias = {}
        for user, ratings in user_item_ratings.items():
            user_avg = sum(ratings.values()) / len(ratings) if ratings else 0.0
            self.user_bias[user] = user_avg - self.global_avg
        
        # Compute item bias: item_avg - global_avg
        self.item_bias = {}
        item_ratings = {}
        for ratings in user_item_ratings.values():
            for item, rating in ratings.items():
                item_ratings.setdefault(item, []).append(rating)
        for item, r_list in item_ratings.items():
            item_avg = sum(r_list) / len(r_list) if r_list else 0.0
            self.item_bias[item] = item_avg - self.global_avg

    def compute_user_similarity(self, user1_id, user2_id):
        """
        Compute adjusted cosine similarity between two users.
        For each item in common, subtract the item's average rating (global_avg + item_bias)
        to remove item bias.
        """
        ratings1 = self.user_item_ratings.get(user1_id, {})
        ratings2 = self.user_item_ratings.get(user2_id, {})
        common_items = set(ratings1.keys()).intersection(ratings2.keys())
        if not common_items:
            return 0.0

        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for item in common_items:
            r1 = ratings1[item]
            r2 = ratings2[item]
            # Compute item's average rating using bias
            item_avg = self.global_avg + self.item_bias.get(item, 0.0)
            adj_r1 = r1 - item_avg
            adj_r2 = r2 - item_avg
            dot_product += adj_r1 * adj_r2
            norm1 += adj_r1 ** 2
            norm2 += adj_r2 ** 2

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))

    def compute_item_similarity(self, item1_id, item2_id):
        """
        Compute adjusted cosine similarity between two items.
        For each user in common, subtract the user's average rating (global_avg + user_bias)
        to remove user bias.
        """
        common_users = [user for user, ratings in self.user_item_ratings.items() 
                        if item1_id in ratings and item2_id in ratings]
        if not common_users:
            return 0.0

        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for user in common_users:
            r1 = self.user_item_ratings[user][item1_id]
            r2 = self.user_item_ratings[user][item2_id]
            # Compute user's average rating using bias
            user_avg = self.global_avg + self.user_bias.get(user, 0.0)
            adj_r1 = r1 - user_avg
            adj_r2 = r2 - user_avg
            dot_product += adj_r1 * adj_r2
            norm1 += adj_r1 ** 2
            norm2 += adj_r2 ** 2

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))

    def predict_user_based_rating(self, target_user_id, target_item_id, k=15):
        """
        Predict the rating for a user-item pair using user-based collaborative filtering
        on the residuals (actual rating - baseline). The baseline is computed as:
          baseline = global_avg + user_bias(target_user) + item_bias(target_item)
        """
        similarities = []
        for other_user_id, ratings in self.user_item_ratings.items():
            if other_user_id == target_user_id:
                continue
            if target_item_id in ratings:
                sim = self.compute_user_similarity(target_user_id, other_user_id)
                if sim > 0:
                    similarities.append((sim, other_user_id))
        similarities.sort(key=lambda x: x[0], reverse=True)
        if len(similarities) > k:
            similarities = similarities[:k]

        weighted_sum = 0.0
        sim_sum = 0.0
        for sim, other_user_id in similarities:
            neighbor_rating = self.user_item_ratings[other_user_id][target_item_id]
            # Baseline for neighbor's rating:
            baseline_neighbor = self.global_avg + self.user_bias.get(other_user_id, 0.0) + self.item_bias.get(target_item_id, 0.0)
            residual = neighbor_rating - baseline_neighbor
            weighted_sum += sim * residual
            sim_sum += abs(sim)
        predicted_residual = 0.0 if sim_sum == 0 else weighted_sum / sim_sum
        # Baseline for target prediction:
        baseline_target = self.global_avg + self.user_bias.get(target_user_id, 0.0) + self.item_bias.get(target_item_id, 0.0)
        return baseline_target + predicted_residual

    def predict_item_based_rating(self, target_user_id, target_item_id, k=15):
        """
        Predict the rating for a user-item pair using item-based collaborative filtering
        on the residuals.
        """
        if target_user_id not in self.user_item_ratings:
            return 0.0

        user_ratings = self.user_item_ratings[target_user_id]
        similarities = []
        for other_item_id, rating in user_ratings.items():
            if other_item_id != target_item_id:
                sim = self.compute_item_similarity(target_item_id, other_item_id)
                if sim > 0:
                    similarities.append((sim, other_item_id))
        similarities.sort(key=lambda x: x[0], reverse=True)
        if len(similarities) > k:
            similarities = similarities[:k]

        weighted_sum = 0.0
        sim_sum = 0.0
        for sim, other_item_id in similarities:
            neighbor_rating = user_ratings[other_item_id]
            # Baseline for neighbor item:
            baseline_neighbor = self.global_avg + self.user_bias.get(target_user_id, 0.0) + self.item_bias.get(other_item_id, 0.0)
            residual = neighbor_rating - baseline_neighbor
            weighted_sum += sim * residual
            sim_sum += abs(sim)
        predicted_residual = 0.0 if sim_sum == 0 else weighted_sum / sim_sum
        # Baseline for target:
        baseline_target = self.global_avg + self.user_bias.get(target_user_id, 0.0) + self.item_bias.get(target_item_id, 0.0)
        return baseline_target + predicted_residual

    def compute_hybrid_prediction(self, user_id, item_id, k_user, k_item, weight_user, weight_item):
        """
        Compute a hybrid prediction by combining user-based and item-based predictions.
        The weights (weight_user and weight_item) must sum to 1.
        """
        if abs(weight_user + weight_item - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.")
        user_based = self.predict_user_based_rating(user_id, item_id, k=k_user)
        item_based = self.predict_item_based_rating(user_id, item_id, k=k_item)
        return (weight_user * user_based) + (weight_item * item_based)

    @staticmethod
    def calculate_rmse(actual, predicted):
        """
        Calculate the Root Mean Square Error (RMSE) between two lists of ratings.
        """
        if len(actual) != len(predicted):
            raise ValueError("Lists must have the same length.")
        squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
        return math.sqrt(sum(squared_errors) / len(actual))

    @staticmethod
    def load_training_data(filepath):
        """
        Load training data from a CSV file.
        Expected format (with or without header): user_id,item_id,rating
        Returns a dictionary: {user_id: {item_id: rating, ...}, ...}
        """
        data = {}
        with open(filepath, newline="") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header and not header[0].isdigit():
                pass  # Skip header row
            else:
                csvfile.seek(0)
                reader = csv.reader(csvfile)
            for row in reader:
                if len(row) != 3:
                    continue
                try:
                    user_id = int(row[0])
                    item_id = int(row[1])
                    rating = float(row[2])
                except ValueError:
                    continue
                data.setdefault(user_id, {})[item_id] = rating
        return data

    @staticmethod
    def load_test_data(filepath):
        """
        Load test data from a CSV file.
        Expected format (with or without header): user_id,item_id,actual_rating
        Returns a list of tuples: (user_id, item_id, actual_rating)
        """
        test_data = []
        with open(filepath, newline="") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header and not header[0].isdigit():
                pass  # Skip header row
            else:
                csvfile.seek(0)
                reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    user_id = int(row[0])
                    item_id = int(row[1])
                    actual_rating = float(row[2])
                except ValueError:
                    continue
                test_data.append((user_id, item_id, actual_rating))
        return test_data

def main():
    # Define file paths
    training_filepath = os.path.join("data", "training.csv")
    test_filepath = os.path.join("data", "test.csv")
    output_filepath = os.path.join("data", "predictions.csv")

    print("Loading training data...")
    training_data = CollaborativeFiltering.load_training_data(training_filepath)
    print("Loading test data...")
    test_data = CollaborativeFiltering.load_test_data(test_filepath)

    # Initialize the model (biases are computed during initialization)
    cf = CollaborativeFiltering(training_data)

    # Hyperparameters for hybrid prediction
    k_user = 15
    k_item = 15

    # Hypothetical average RMSE values (for weight calculation)
    average_rmse_user = 0.4
    average_rmse_item = 0.6
    weight_user = (1.0 / average_rmse_user) / ((1.0 / average_rmse_user) + (1.0 / average_rmse_item))
    weight_item = (1.0 / average_rmse_item) / ((1.0 / average_rmse_user) + (1.0 / average_rmse_item))

    predictions = []
    actual_ratings = []
    predicted_ratings = []

    print("Predicting hybrid ratings...")
    for user_id, item_id, actual_rating in tqdm(test_data, desc="Predicting", unit="pair"):
        predicted_rating = cf.compute_hybrid_prediction(user_id, item_id, k_user, k_item, weight_user, weight_item)
        predictions.append((user_id, item_id, predicted_rating, actual_rating))
        predicted_ratings.append(predicted_rating)
        actual_ratings.append(actual_rating)

    total_rmse = cf.calculate_rmse(actual_ratings, predicted_ratings)
    print(f"Total RMSE: {total_rmse}")

    # Write predictions and overall RMSE to an output CSV file
    with open(output_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["user_id", "item_id", "predicted_rating", "actual_rating"])
        for row in predictions:
            writer.writerow(row)
        writer.writerow(["Total_RMSE", "", "", total_rmse])
    print(f"Predictions and RMSE written to {output_filepath}")

if __name__ == "__main__":
    main()

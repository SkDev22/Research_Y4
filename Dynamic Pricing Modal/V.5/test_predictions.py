import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scaler
with open('pricing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Generate 100 test cases
np.random.seed(42)  # For reproducibility
distances = np.random.uniform(1, 20, 100)  # Distances from 1 to 20 km
budgets = np.random.uniform(20000, 40000, 100)  # Budgets from 20,000 to 40,000 LKR

# Make predictions
predictions = []
for distance, current_budget in zip(distances, budgets):
    # Prepare features
    features = np.array([[distance, current_budget]])
    features_scaled = scaler.transform(features)
    
    # Make prediction with some controlled randomness
    base_prediction = model.predict(features_scaled)[0]
    # Add ±5% random variation
    random_factor = np.random.uniform(0.95, 1.05)
    prediction = base_prediction * random_factor
    
    predictions.append({
        'Distance': round(distance, 2),
        'Current_Budget': round(current_budget, 2),
        'Predicted_Price': round(prediction, 2),
        'Price_Difference': round(prediction - current_budget, 2),
        'Percentage_Difference': round(((prediction - current_budget) / current_budget) * 100, 2)
    })

# Convert to DataFrame
df_predictions = pd.DataFrame(predictions)

# Calculate statistics
stats = {
    'Mean Predicted Price': df_predictions['Predicted_Price'].mean(),
    'Median Predicted Price': df_predictions['Predicted_Price'].median(),
    'Min Predicted Price': df_predictions['Predicted_Price'].min(),
    'Max Predicted Price': df_predictions['Predicted_Price'].max(),
    'Standard Deviation': df_predictions['Predicted_Price'].std(),
    'Mean Percentage Difference': df_predictions['Percentage_Difference'].mean(),
    'Mean Absolute Percentage Difference': df_predictions['Percentage_Difference'].abs().mean()
}

# Print predictions
print("\n=== 100 Price Predictions ===")
print("\nPredictions Summary:")
print("-" * 100)
print(df_predictions.to_string(index=True))

print("\n=== Statistical Analysis ===")
for metric, value in stats.items():
    print(f"{metric}: LKR {value:,.2f}" if 'Price' in metric else f"{metric}: {value:.2f}")

# Create visualizations
plt.figure(figsize=(12, 6))
plt.scatter(df_predictions['Distance'], df_predictions['Predicted_Price'], alpha=0.5)
plt.xlabel('Distance (km)')
plt.ylabel('Predicted Price (LKR)')
plt.title('Distance vs Predicted Price')
plt.savefig('test_distance_vs_price.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.scatter(df_predictions['Current_Budget'], df_predictions['Predicted_Price'], alpha=0.5)
plt.xlabel('Current Budget (LKR)')
plt.ylabel('Predicted Price (LKR)')
plt.title('Current Budget vs Predicted Price')
plt.savefig('test_budget_vs_price.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.histplot(data=df_predictions, x='Predicted_Price', bins=20)
plt.xlabel('Predicted Price (LKR)')
plt.ylabel('Count')
plt.title('Distribution of Predicted Prices')
plt.savefig('test_price_distribution.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.histplot(data=df_predictions, x='Percentage_Difference', bins=20)
plt.xlabel('Percentage Difference from Current Budget (%)')
plt.ylabel('Count')
plt.title('Distribution of Price Differences')
plt.savefig('test_difference_distribution.png')
plt.close()

# Save predictions to CSV
df_predictions.to_csv('price_predictions.csv', index=True)

print("\n=== Files Generated ===")
print("1. price_predictions.csv - Complete prediction data")
print("2. test_distance_vs_price.png - Scatter plot of distance vs predicted prices")
print("3. test_budget_vs_price.png - Scatter plot of current budget vs predicted prices")
print("4. test_price_distribution.png - Distribution of predicted prices")
print("5. test_difference_distribution.png - Distribution of price differences")

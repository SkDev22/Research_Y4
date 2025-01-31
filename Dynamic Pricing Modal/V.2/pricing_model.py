import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from datetime import datetime

# Enhanced sample data with more realistic factors
def generate_sample_data(n_samples=100):
    np.random.seed(42)
    
    # Distance (km) - More properties closer to uni (using exponential distribution)
    distances = np.random.exponential(scale=5, size=n_samples)
    distances = np.clip(distances, 0.2, 20)  # Limit distance range
    
    # Amenities Score (1-10)
    # 1-3: Basic (Fan, Shared Bathroom)
    # 4-6: Standard (AC, Attached Bathroom)
    # 7-8: Premium (Study Area, Kitchen)
    # 9-10: Luxury (Gym, Security, Laundry)
    amenities_scores = np.random.choice(
        np.arange(1, 11),
        size=n_samples,
        p=[0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.08, 0.07, 0.03, 0.02]  # More basic/standard than premium
    )
    
    # Seasonal Demand (1-10)
    # Higher during orientation (Jan/Feb)
    # Lower during graduation (June/July)
    # Medium-high during events
    def get_seasonal_demand(month):
        if month in [1, 2]:  # Orientation season
            return np.random.uniform(8, 10)
        elif month in [6, 7]:  # Graduation season
            return np.random.uniform(3, 5)
        elif month in [3, 4, 8, 9, 10]:  # Event seasons
            return np.random.uniform(6, 8)
        else:
            return np.random.uniform(5, 7)
    
    current_month = datetime.now().month
    seasonal_demands = np.array([get_seasonal_demand(current_month) for _ in range(n_samples)])
    
    # Event Proximity Score (1-10)
    # Higher score for properties closer to event venues
    event_proximity = np.where(
        distances < 2,  # Very close to uni/events
        np.random.uniform(8, 10, n_samples),
        np.where(
            distances < 5,  # Moderately close
            np.random.uniform(5, 7, n_samples),
            np.random.uniform(2, 4, n_samples)  # Far from events
        )
    )
    
    # Calculate base price (LKR)
    # Distance has inverse relationship with price
    # Closer properties get premium pricing
    base_price = 10000 + (
        (10 / (distances + 0.1)) * 2000 +  # Distance factor (inverse relationship, max +20000)
        (amenities_scores * 1500) +        # Amenities premium (max +15000)
        (seasonal_demands * 500) +         # Seasonal demand factor (max +5000)
        (event_proximity * 500)            # Event proximity premium (max +5000)
    )
    
    # Add some random variation (±5%)
    base_price *= np.random.uniform(0.95, 1.05, n_samples)
    
    # Clip prices to ensure they stay within affordable range
    base_price = np.clip(base_price, 10000, 55000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Distance': distances,
        'Amenities_Score': amenities_scores,
        'Seasonal_Demand': seasonal_demands,
        'Event_Proximity': event_proximity,
        'Price': base_price
    })
    
    return df

def train_model():
    # Generate enhanced sample data
    df = generate_sample_data(200)  # More samples for better training
    
    # Prepare features and target
    X = df[['Distance', 'Amenities_Score', 'Seasonal_Demand', 'Event_Proximity']]
    y = df['Price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model (better for complex relationships)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return r2, mae, model, scaler, y_test, y_pred, df

def predict_price(distance, amenities_score, seasonal_demand, event_proximity, model, scaler):
    # Prepare features
    features = np.array([[distance, amenities_score, seasonal_demand, event_proximity]])
    features_scaled = scaler.transform(features)
    
    # Predict price
    predicted_price = model.predict(features_scaled)[0]
    
    return predicted_price

def main():
    print("Training enhanced pricing model...")
    r2, mae, model, scaler, y_test, y_pred, df = train_model()
    
    print("\nModel Performance:")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: LKR {mae:.2f}")
    
    # Calculate average price difference
    avg_diff_percent = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
    print(f"\nAverage price difference: {avg_diff_percent:.1f}%")
    
    # Example predictions for different scenarios
    print("\nExample Predictions:")
    print("-" * 50)
    
    scenarios = [
        {
            'name': 'Premium Close Property (Orientation Season)',
            'distance': 0.5,
            'amenities': 9,
            'seasonal_demand': 9,
            'event_proximity': 9
        },
        {
            'name': 'Standard Mid-Range Property',
            'distance': 5,
            'amenities': 6,
            'seasonal_demand': 6,
            'event_proximity': 5
        },
        {
            'name': 'Budget Distant Property (Off-Season)',
            'distance': 15,
            'amenities': 3,
            'seasonal_demand': 4,
            'event_proximity': 2
        }
    ]
    
    for scenario in scenarios:
        price = predict_price(
            scenario['distance'],
            scenario['amenities'],
            scenario['seasonal_demand'],
            scenario['event_proximity'],
            model,
            scaler
        )
        
        print(f"\nScenario: {scenario['name']}")
        print(f"Distance: {scenario['distance']}km")
        print(f"Amenities Score: {scenario['amenities']}/10")
        print(f"Seasonal Demand: {scenario['seasonal_demand']}/10")
        print(f"Event Proximity: {scenario['event_proximity']}/10")
        print(f"Predicted Price: LKR {price:,.2f}")
    
    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Distance'], df['Price'], alpha=0.5)
    plt.xlabel('Distance to University (km)')
    plt.ylabel('Price (LKR)')
    plt.title('Distance vs Price (Inverse Relationship)')
    plt.savefig('distance_vs_price.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Amenities_Score'], df['Price'], alpha=0.5)
    plt.xlabel('Amenities Score')
    plt.ylabel('Price (LKR)')
    plt.title('Amenities Score vs Price')
    plt.savefig('amenities_vs_price.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Seasonal_Demand'], df['Price'], alpha=0.5)
    plt.xlabel('Seasonal Demand Score')
    plt.ylabel('Price (LKR)')
    plt.title('Seasonal Demand vs Price')
    plt.savefig('seasonal_vs_price.png')
    plt.close()
    
    print("\nVisualization graphs have been saved:")
    print("1. distance_vs_price.png - Shows inverse relationship between distance and price")
    print("2. amenities_vs_price.png - Shows impact of amenities on price")
    print("3. seasonal_vs_price.png - Shows seasonal demand effects")
    
    # Save the model and scaler
    joblib.dump(model, 'pricing_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    with open('pricing_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModel files have been saved:")
    print("1. pricing_model.joblib & pricing_model.pkl - The trained model")
    print("2. scaler.joblib & scaler.pkl - The fitted scaler")

if __name__ == "__main__":
    main()

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
from datetime import datetime, timedelta

class BoardingHouseAmenities:
    def __init__(self):
        # Basic amenities
        self.has_wifi = False
        self.wifi_speed = 0  # Speed in Mbps
        self.has_generator = False
        self.generator_capacity = 0  # KVA
        self.water_tank_capacity = 0  # Liters
        
        # Cultural amenities
        self.has_prayer_room = False
        self.separate_veg_kitchen = False
        self.distance_to_temple = 0.0  # km
        self.distance_to_mosque = 0.0  # km
        self.distance_to_church = 0.0  # km
        
        # Environmental features
        self.building_age = 0  # years
        self.flood_risk = 0  # 0-10 (0 = no risk)
        self.noise_level = 0  # 0-10 (0 = very quiet)
        self.ventilation_score = 0  # 0-10
        self.has_pest_control = False
        self.pest_control_frequency = 0  # months
        
        # Common areas
        self.has_common_room = False
        self.common_room_size = 0  # sq ft
        self.has_study_hall = False
        self.study_hall_capacity = 0  # persons
        
        # Room features
        self.bed_type = None  # single/double
        self.has_cupboard = False         # Storage is important
        self.has_study_table = False      # Essential for students
        self.has_chair = False
        self.has_balcony = False         # Premium feature in SL
        self.room_type = None            # 'single', 'shared-2', 'shared-4'
        
        # Utilities included
        self.water_included = False      # Common utility concern
        self.electricity_included = False # Important for budget planning
        
        # Additional features
        self.meals_provided = False      # Common in SL boarding houses
        self.cleaning_service = False
        self.gender_specific = None      # 'male', 'female', 'any'
        self.has_warden = False         # Common in SL, adds security
        self.has_visitor_area = False   # Important for meeting friends/family
        
    def calculate_amenity_score(self):
        score = 0
        
        # Essential amenities (higher weightage)
        if self.has_wifi: score += 1.5        # Essential for studies
        if self.has_attached_bathroom: score += 2
        if self.has_shared_bathroom: score += 0.5
        if self.has_ac: score += 1.5          # Luxury in SL
        if self.has_fan: score += 1           # Essential in SL climate
        
        # Room features
        if self.has_cupboard: score += 0.5
        if self.has_study_table: score += 0.8
        if self.has_chair: score += 0.3
        if self.has_balcony: score += 0.5
        
        # Parking
        if self.has_parking: score += 0.5
        if self.has_bike_parking: score += 0.8  # More relevant for students
        
        # Kitchen facilities
        if self.has_kitchen: score += 1.5
        if self.has_shared_kitchen: score += 0.8
        
        # Security and convenience
        if self.has_cctv: score += 0.8
        if self.has_backup_power: score += 1    # Important during power cuts
        if self.has_warden: score += 0.8
        if self.has_visitor_area: score += 0.5
        
        # Utilities included (significant value)
        if self.water_included: score += 1
        if self.electricity_included: score += 1.5
        
        # Additional services
        if self.meals_provided: score += 2      # Very valuable in SL
        if self.cleaning_service: score += 0.8
        if self.has_laundry: score += 0.8
        
        # Room type adjustments
        if self.room_type == 'single': score += 1.5
        elif self.room_type == 'shared-2': score += 0.8
        elif self.room_type == 'shared-4': score += 0.3
        
        # Cultural amenities scoring
        if self.has_prayer_room:
            score += 0.8
        if self.separate_veg_kitchen:
            score += 0.7
        
        # Calculate religious place proximity score
        min_religious_distance = min(
            self.distance_to_temple,
            self.distance_to_mosque,
            self.distance_to_church,
            default=5.0
        )
        if min_religious_distance < 0.5:
            score += 1.0
        elif min_religious_distance < 1.0:
            score += 0.5
        
        # Environmental scoring
        score += (10 - self.building_age) / 20  # Newer buildings score higher
        score += (10 - self.flood_risk) / 10
        score += (10 - self.noise_level) / 10
        score += self.ventilation_score / 10
        
        if self.has_pest_control:
            score += 0.5 + (12 - self.pest_control_frequency) / 24  # More frequent = better
        
        # Utilities scoring
        if self.wifi_speed > 100:
            score += 1.5
        elif self.wifi_speed > 50:
            score += 1.0
        elif self.wifi_speed > 20:
            score += 0.5
        
        if self.has_generator:
            score += 0.5 + (self.generator_capacity / 10)  # Higher capacity = better
        
        if self.water_tank_capacity > 2000:
            score += 1.0
        elif self.water_tank_capacity > 1000:
            score += 0.5
        
        # Common areas scoring
        if self.has_common_room:
            score += 0.5 + (self.common_room_size / 500)  # Larger rooms score higher
        if self.has_study_hall:
            score += 0.5 + (self.study_hall_capacity / 20)  # Higher capacity = better
        
        return min(score, 10)  # Cap at 10
        
    def get_amenities_description(self):
        amenities = []
        if self.has_wifi: amenities.append("WiFi")
        if self.has_attached_bathroom: amenities.append("Attached Bathroom")
        elif self.has_shared_bathroom: amenities.append("Shared Bathroom")
        if self.has_ac: amenities.append("Air Conditioning")
        if self.has_fan: amenities.append("Fan")
        if self.has_parking: amenities.append("Car Parking")
        if self.has_bike_parking: amenities.append("Bike Parking")
        if self.has_kitchen: amenities.append("Private Kitchen")
        elif self.has_shared_kitchen: amenities.append("Shared Kitchen")
        if self.has_laundry: amenities.append("Laundry Facility")
        if self.has_study_area: amenities.append("Study Area")
        if self.has_cctv: amenities.append("CCTV Security")
        if self.has_backup_power: amenities.append("Backup Power")
        if self.has_cupboard: amenities.append("Cupboard")
        if self.has_study_table: amenities.append("Study Table")
        if self.has_balcony: amenities.append("Balcony")
        if self.water_included: amenities.append("Water Included")
        if self.electricity_included: amenities.append("Electricity Included")
        if self.meals_provided: amenities.append("Meals Available")
        if self.cleaning_service: amenities.append("Cleaning Service")
        if self.has_warden: amenities.append("Warden Service")
        if self.has_visitor_area: amenities.append("Visitor Area")
        if self.room_type: amenities.append(f"{self.room_type.title()} Room")
        if self.gender_specific: amenities.append(f"{self.gender_specific.title()} Only")
        if self.has_prayer_room: amenities.append("Prayer Room")
        if self.separate_veg_kitchen: amenities.append("Separate Veg Kitchen")
        if self.has_common_room: amenities.append("Common Room")
        if self.has_study_hall: amenities.append("Study Hall")
        return amenities

class LocationFeatures:
    def __init__(self):
        self.distance_to_uni = 0.0        # in km
        self.distance_to_food = 0.0       # in km
        self.distance_to_transport = 0.0   # in km
        self.is_main_road = False         # True if on main road
        self.is_residential_area = False   # True if in quiet residential area
        self.safety_score = 0             # 1-10 scale
        
    def calculate_location_score(self):
        score = 0
        # Transport accessibility
        if self.distance_to_transport < 0.5:
            score += 2
        elif self.distance_to_transport < 1:
            score += 1
        
        # Food accessibility
        if self.distance_to_food < 0.5:
            score += 1.5
        elif self.distance_to_food < 1:
            score += 0.8
        
        # Area type bonus
        if self.is_residential_area:
            score += 1  # Quiet area premium
        if self.is_main_road:
            score += 0.5  # Better accessibility
        
        # Safety score (normalized)
        score += (self.safety_score / 10) * 2
        
        return min(score, 10)

class ReviewMetrics:
    def __init__(self):
        self.overall_rating = 0.0         # 1-5 scale
        self.num_reviews = 0
        self.landlord_response_time = 0   # hours
        self.maintenance_rating = 0.0      # 1-5 scale
        self.cleanliness_rating = 0.0     # 1-5 scale
        self.value_rating = 0.0           # 1-5 scale
        
    def calculate_review_score(self):
        if self.num_reviews == 0:
            return 5.0  # Default score for new listings
        
        score = 0
        # Overall rating (weighted heavily)
        score += self.overall_rating * 2
        
        # Response time score
        response_score = 0
        if self.landlord_response_time < 1:
            response_score = 1.0
        elif self.landlord_response_time < 4:
            response_score = 0.8
        elif self.landlord_response_time < 12:
            response_score = 0.5
        score += response_score
        
        # Maintenance and cleanliness
        score += (self.maintenance_rating + self.cleanliness_rating) / 2
        
        # Value rating
        score += self.value_rating
        
        # Normalize to 10-point scale
        return min((score / 5) * 10, 10)

def get_seasonal_factor(month):
    # Seasonal factors adjusted for Sri Lankan academic calendar
    seasons = {
        # First Semester Start (High demand)
        1: 1.15,  # January - New semester registration
        2: 1.15,  # February - Late registrations
        
        # Mid First Semester (Normal demand)
        3: 1.0,   # March
        4: 1.0,   # April
        
        # First Semester End / Festivals (Mixed demand)
        5: 0.95,  # May - Vesak season
        
        # Inter-Semester Break (Low demand)
        6: 0.85,  # June - Many students return home
        7: 0.85,  # July - Break continues
        
        # Second Semester Start (High demand)
        8: 1.12,  # August - Second semester begins
        9: 1.10,  # September
        
        # Festival Season (Mixed demand)
        10: 0.95, # October - Deepavali season
        11: 1.0,  # November
        
        # Year-end Examinations (Stable demand)
        12: 1.0   # December
    }
    return seasons.get(month, 1.0)

def generate_sample_data(n_samples=100):
    np.random.seed(42)
    
    # Distance (km) - More properties closer to uni
    distances = np.random.exponential(scale=5, size=n_samples)
    distances = np.clip(distances, 0.2, 20)
    
    # Generate amenities for each property
    amenities_list = []
    location_list = []
    review_list = []
    
    for _ in range(n_samples):
        # Generate amenities
        amenities = BoardingHouseAmenities()
        prob_factor = 1 - (distances[_] / 20)
        
        amenities.has_wifi = np.random.random() < 0.8  # Most have WiFi
        amenities.wifi_speed = np.random.choice([20, 50, 100, 200])
        amenities.has_generator = np.random.random() < 0.6
        amenities.generator_capacity = np.random.randint(2, 10) if amenities.has_generator else 0
        amenities.water_tank_capacity = np.random.randint(500, 3000)
        
        amenities.has_prayer_room = np.random.random() < 0.3
        amenities.separate_veg_kitchen = np.random.random() < 0.4
        amenities.distance_to_temple = np.random.uniform(0.1, 3.0)
        amenities.distance_to_mosque = np.random.uniform(0.1, 3.0)
        amenities.distance_to_church = np.random.uniform(0.1, 3.0)
        
        amenities.building_age = np.random.randint(0, 20)
        amenities.flood_risk = np.random.randint(0, 10)
        amenities.noise_level = np.random.randint(0, 10)
        amenities.ventilation_score = np.random.randint(5, 10)
        amenities.has_pest_control = np.random.random() < 0.7
        amenities.pest_control_frequency = np.random.randint(1, 12)
        
        amenities.has_common_room = np.random.random() < 0.5
        amenities.common_room_size = np.random.randint(200, 1000) if amenities.has_common_room else 0
        amenities.has_study_hall = np.random.random() < 0.4
        amenities.study_hall_capacity = np.random.randint(10, 30) if amenities.has_study_hall else 0
        
        amenities.has_attached_bathroom = np.random.random() < 0.6 * (1 + prob_factor)
        amenities.has_shared_bathroom = np.random.random() < 0.4 * (1 + prob_factor)
        amenities.has_ac = np.random.random() < 0.4 * (1 + prob_factor)
        amenities.has_fan = np.random.random() < 0.9  # Most have fans
        amenities.has_parking = np.random.random() < 0.5 * (1 + prob_factor)
        amenities.has_bike_parking = np.random.random() < 0.7 * (1 + prob_factor)
        amenities.has_kitchen = np.random.random() < 0.4 * (1 + prob_factor)
        amenities.has_shared_kitchen = np.random.random() < 0.3 * (1 + prob_factor)
        amenities.has_laundry = np.random.random() < 0.3 * (1 + prob_factor)
        amenities.has_study_area = np.random.random() < 0.4 * (1 + prob_factor)
        amenities.has_cctv = np.random.random() < 0.3 * (1 + prob_factor)
        amenities.has_backup_power = np.random.random() < 0.3 * (1 + prob_factor)
        amenities.meals_provided = np.random.random() < 0.2 * (1 + prob_factor)
        amenities.cleaning_service = np.random.random() < 0.2 * (1 + prob_factor)
        amenities.has_cupboard = np.random.random() < 0.6 * (1 + prob_factor)
        amenities.has_study_table = np.random.random() < 0.7 * (1 + prob_factor)
        amenities.has_chair = np.random.random() < 0.8 * (1 + prob_factor)
        amenities.has_balcony = np.random.random() < 0.4 * (1 + prob_factor)
        amenities.water_included = np.random.random() < 0.5 * (1 + prob_factor)
        amenities.electricity_included = np.random.random() < 0.4 * (1 + prob_factor)
        amenities.has_warden = np.random.random() < 0.3 * (1 + prob_factor)
        amenities.has_visitor_area = np.random.random() < 0.2 * (1 + prob_factor)
        amenities.room_type = np.random.choice(['single', 'shared-2', 'shared-4'], p=[0.4, 0.3, 0.3])
        amenities.gender_specific = np.random.choice(['male', 'female', 'any'], p=[0.4, 0.3, 0.3])
        
        # Generate location features
        location = LocationFeatures()
        location.distance_to_uni = distances[_]
        location.distance_to_food = np.random.uniform(0.1, 2.0)
        location.distance_to_transport = np.random.uniform(0.1, 1.5)
        location.is_main_road = np.random.random() < 0.4
        location.is_residential_area = np.random.random() < 0.6
        location.safety_score = np.random.randint(5, 10)
        
        # Generate review metrics
        reviews = ReviewMetrics()
        reviews.overall_rating = np.random.uniform(3.5, 5.0)
        reviews.num_reviews = np.random.randint(0, 100)
        reviews.landlord_response_time = np.random.uniform(1, 24)
        reviews.maintenance_rating = np.random.uniform(3.0, 5.0)
        reviews.cleanliness_rating = np.random.uniform(3.0, 5.0)
        reviews.value_rating = np.random.uniform(3.0, 5.0)
        
        amenities_list.append(amenities)
        location_list.append(location)
        review_list.append(reviews)
    
    # Calculate scores
    amenities_scores = np.array([a.calculate_amenity_score() for a in amenities_list])
    location_scores = np.array([l.calculate_location_score() for l in location_list])
    review_scores = np.array([r.calculate_review_score() for r in review_list])
    
    # Current month for seasonal factors
    current_month = datetime.now().month
    seasonal_demands = np.array([get_seasonal_factor(current_month) for _ in range(n_samples)])
    
    # Event Proximity Score (1-10)
    event_proximity = np.where(
        distances < 2,
        np.random.uniform(8, 10, n_samples),
        np.where(
            distances < 5,
            np.random.uniform(5, 7, n_samples),
            np.random.uniform(2, 4, n_samples)
        )
    )
    
    # Calculate base price (LKR)
    base_price = 10000 + (
        (10 / (distances + 0.1)) * 2000 +     # Distance factor
        (amenities_scores * 1500) +           # Amenities premium
        (location_scores * 1000) +            # Location premium
        (review_scores * 500) +               # Review premium
        (event_proximity * 500)               # Event proximity premium
    )
    
    # Apply seasonal factor
    base_price *= seasonal_demands
    
    # Add some random variation (±5%)
    base_price *= np.random.uniform(0.95, 1.05, n_samples)
    
    # Clip prices to ensure they stay within affordable range
    base_price = np.clip(base_price, 10000, 55000)
    
    # Calculate last month and next month prices
    last_month = (datetime.now() - timedelta(days=30)).month
    next_month = (datetime.now() + timedelta(days=30)).month
    
    last_month_prices = base_price * (get_seasonal_factor(last_month) / get_seasonal_factor(current_month))
    next_month_prices = base_price * (get_seasonal_factor(next_month) / get_seasonal_factor(current_month))
    
    # Create DataFrame with all features
    df = pd.DataFrame({
        'Distance': distances,
        'Amenities_Score': amenities_scores,
        'Location_Score': location_scores,
        'Review_Score': review_scores,
        'Seasonal_Demand': seasonal_demands,
        'Event_Proximity': event_proximity,
        'Current_Price': base_price,
        'Last_Month_Price': last_month_prices,
        'Next_Month_Price': next_month_prices
    })
    
    return df, amenities_list, location_list, review_list

def predict_monthly_prices(distance, amenities, model, scaler, location=None, reviews=None):
    # Calculate amenity score
    amenity_score = amenities.calculate_amenity_score()
    
    # Get location and review scores if available
    location_score = location.calculate_location_score() if location else 5.0
    review_score = reviews.calculate_review_score() if reviews else 5.0
    
    # Get current month
    current_month = datetime.now().month
    last_month = (datetime.now() - timedelta(days=30)).month
    next_month = (datetime.now() + timedelta(days=30)).month
    
    # Calculate event proximity based on distance
    event_proximity = 9 if distance < 2 else (6 if distance < 5 else 3)
    
    # Prepare features for current month
    features = np.array([[
        distance,
        amenity_score,
        get_seasonal_factor(current_month),
        event_proximity,
        location_score / 10,  # Normalize to 0-1
        review_score / 10     # Normalize to 0-1
    ]])
    features_scaled = scaler.transform(features)
    
    # Predict current month price
    current_price = model.predict(features_scaled)[0]
    
    # Apply annual inflation rate (assume 6.5% for Sri Lanka)
    inflation_factor = 1.065  # Can be adjusted based on current inflation rate
    
    # Calculate prices for adjacent months using seasonal factors
    last_month_price = current_price * (get_seasonal_factor(last_month) / get_seasonal_factor(current_month))
    next_month_price = current_price * (get_seasonal_factor(next_month) / get_seasonal_factor(current_month))
    
    # Project prices for next year
    next_year_estimate = current_price * inflation_factor
    
    return {
        'last_month': last_month_price,
        'current': current_price,
        'next_month': next_month_price,
        'next_year_estimate': next_year_estimate,
        'amenities': amenities.get_amenities_description(),
        'location_score': location_score if location else None,
        'review_score': review_score if reviews else None
    }

def main():
    print("Training enhanced pricing model...")
    
    # Generate sample data with all features
    df, amenities_list, location_list, review_list = generate_sample_data(200)
    
    # Prepare features and target
    X = df[['Distance', 'Amenities_Score', 'Location_Score', 'Review_Score', 'Seasonal_Demand', 'Event_Proximity']]
    y = df['Current_Price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: LKR {mae:.2f}")
    
    # Example predictions for different scenarios
    print("\nExample Predictions:")
    print("-" * 50)
    
    scenarios = [
        {
            'name': 'Premium Close Property',
            'distance': 0.5,
            'amenities': {
                'wifi': True,
                'attached_bathroom': True,
                'ac': True,
                'fan': True,
                'parking': True,
                'kitchen': True,
                'study_area': True,
                'cctv': True,
                'backup_power': True,
                'meals': False,
                'cleaning': True,
                'cupboard': True,
                'study_table': True,
                'chair': True,
                'balcony': True,
                'water_included': True,
                'electricity_included': True,
                'warden': True,
                'visitor_area': True,
                'room_type': 'single',
                'gender_specific': 'any',
                'has_prayer_room': True,
                'separate_veg_kitchen': True,
                'distance_to_temple': 0.5,
                'distance_to_mosque': 1.0,
                'distance_to_church': 1.5,
                'building_age': 5,
                'flood_risk': 2,
                'noise_level': 3,
                'ventilation_score': 8,
                'has_pest_control': True,
                'pest_control_frequency': 3,
                'has_common_room': True,
                'common_room_size': 500,
                'has_study_hall': True,
                'study_hall_capacity': 20,
                'wifi_speed': 100,
                'has_generator': True,
                'generator_capacity': 5,
                'water_tank_capacity': 2000
            },
            'location': {
                'distance_to_uni': 0.5,
                'distance_to_food': 0.2,
                'distance_to_transport': 0.1,
                'is_main_road': True,
                'is_residential_area': False,
                'safety_score': 9
            },
            'reviews': {
                'overall_rating': 4.8,
                'num_reviews': 50,
                'landlord_response_time': 1,
                'maintenance_rating': 4.5,
                'cleanliness_rating': 4.8,
                'value_rating': 4.7
            }
        },
        {
            'name': 'Standard Mid-Range Property',
            'distance': 5,
            'amenities': {
                'wifi': True,
                'attached_bathroom': True,
                'ac': False,
                'fan': True,
                'parking': True,
                'kitchen': False,
                'study_area': True,
                'cctv': False,
                'backup_power': False,
                'meals': False,
                'cleaning': False,
                'cupboard': True,
                'study_table': True,
                'chair': True,
                'balcony': False,
                'water_included': False,
                'electricity_included': False,
                'warden': False,
                'visitor_area': False,
                'room_type': 'shared-2',
                'gender_specific': 'male',
                'has_prayer_room': False,
                'separate_veg_kitchen': False,
                'distance_to_temple': 1.0,
                'distance_to_mosque': 1.5,
                'distance_to_church': 2.0,
                'building_age': 10,
                'flood_risk': 5,
                'noise_level': 5,
                'ventilation_score': 6,
                'has_pest_control': False,
                'pest_control_frequency': 6,
                'has_common_room': False,
                'common_room_size': 0,
                'has_study_hall': False,
                'study_hall_capacity': 0,
                'wifi_speed': 50,
                'has_generator': False,
                'generator_capacity': 0,
                'water_tank_capacity': 1000
            },
            'location': {
                'distance_to_uni': 2,
                'distance_to_food': 0.5,
                'distance_to_transport': 0.3,
                'is_main_road': False,
                'is_residential_area': True,
                'safety_score': 8
            },
            'reviews': {
                'overall_rating': 4.2,
                'num_reviews': 20,
                'landlord_response_time': 2,
                'maintenance_rating': 4.0,
                'cleanliness_rating': 4.2,
                'value_rating': 4.1
            }
        },
        {
            'name': 'Budget Distant Property',
            'distance': 15,
            'amenities': {
                'wifi': True,
                'attached_bathroom': False,
                'ac': False,
                'fan': True,
                'parking': False,
                'kitchen': False,
                'study_area': False,
                'cctv': False,
                'backup_power': False,
                'meals': False,
                'cleaning': False,
                'cupboard': False,
                'study_table': False,
                'chair': False,
                'balcony': False,
                'water_included': False,
                'electricity_included': False,
                'warden': False,
                'visitor_area': False,
                'room_type': 'shared-4',
                'gender_specific': 'female',
                'has_prayer_room': False,
                'separate_veg_kitchen': False,
                'distance_to_temple': 2.0,
                'distance_to_mosque': 2.5,
                'distance_to_church': 3.0,
                'building_age': 15,
                'flood_risk': 8,
                'noise_level': 7,
                'ventilation_score': 4,
                'has_pest_control': False,
                'pest_control_frequency': 12,
                'has_common_room': False,
                'common_room_size': 0,
                'has_study_hall': False,
                'study_hall_capacity': 0,
                'wifi_speed': 20,
                'has_generator': False,
                'generator_capacity': 0,
                'water_tank_capacity': 500
            },
            'location': {
                'distance_to_uni': 10,
                'distance_to_food': 1,
                'distance_to_transport': 0.5,
                'is_main_road': False,
                'is_residential_area': False,
                'safety_score': 6
            },
            'reviews': {
                'overall_rating': 3.8,
                'num_reviews': 10,
                'landlord_response_time': 4,
                'maintenance_rating': 3.5,
                'cleanliness_rating': 3.8,
                'value_rating': 3.7
            }
        }
    ]
    
    for scenario in scenarios:
        # Create amenities object
        amenities = BoardingHouseAmenities()
        amenities.has_wifi = scenario['amenities']['wifi']
        amenities.wifi_speed = scenario['amenities']['wifi_speed']
        amenities.has_generator = scenario['amenities']['has_generator']
        amenities.generator_capacity = scenario['amenities']['generator_capacity']
        amenities.water_tank_capacity = scenario['amenities']['water_tank_capacity']
        
        amenities.has_prayer_room = scenario['amenities']['has_prayer_room']
        amenities.separate_veg_kitchen = scenario['amenities']['separate_veg_kitchen']
        amenities.distance_to_temple = scenario['amenities']['distance_to_temple']
        amenities.distance_to_mosque = scenario['amenities']['distance_to_mosque']
        amenities.distance_to_church = scenario['amenities']['distance_to_church']
        
        amenities.building_age = scenario['amenities']['building_age']
        amenities.flood_risk = scenario['amenities']['flood_risk']
        amenities.noise_level = scenario['amenities']['noise_level']
        amenities.ventilation_score = scenario['amenities']['ventilation_score']
        amenities.has_pest_control = scenario['amenities']['has_pest_control']
        amenities.pest_control_frequency = scenario['amenities']['pest_control_frequency']
        
        amenities.has_common_room = scenario['amenities']['has_common_room']
        amenities.common_room_size = scenario['amenities']['common_room_size']
        amenities.has_study_hall = scenario['amenities']['has_study_hall']
        amenities.study_hall_capacity = scenario['amenities']['study_hall_capacity']
        
        amenities.has_attached_bathroom = scenario['amenities']['attached_bathroom']
        amenities.has_shared_bathroom = not amenities.has_attached_bathroom
        amenities.has_ac = scenario['amenities']['ac']
        amenities.has_fan = scenario['amenities']['fan']
        amenities.has_parking = scenario['amenities']['parking']
        amenities.has_bike_parking = scenario['amenities']['parking']
        amenities.has_kitchen = scenario['amenities']['kitchen']
        amenities.has_shared_kitchen = not amenities.has_kitchen
        amenities.has_laundry = scenario['amenities']['study_area']
        amenities.has_study_area = scenario['amenities']['study_area']
        amenities.has_cctv = scenario['amenities']['cctv']
        amenities.has_backup_power = scenario['amenities']['backup_power']
        amenities.meals_provided = scenario['amenities']['meals']
        amenities.cleaning_service = scenario['amenities']['cleaning']
        amenities.has_cupboard = scenario['amenities']['cupboard']
        amenities.has_study_table = scenario['amenities']['study_table']
        amenities.has_chair = scenario['amenities']['chair']
        amenities.has_balcony = scenario['amenities']['balcony']
        amenities.water_included = scenario['amenities']['water_included']
        amenities.electricity_included = scenario['amenities']['electricity_included']
        amenities.has_warden = scenario['amenities']['warden']
        amenities.has_visitor_area = scenario['amenities']['visitor_area']
        amenities.room_type = scenario['amenities']['room_type']
        amenities.gender_specific = scenario['amenities']['gender_specific']
        
        # Create location object
        location = LocationFeatures()
        location.distance_to_uni = scenario['location']['distance_to_uni']
        location.distance_to_food = scenario['location']['distance_to_food']
        location.distance_to_transport = scenario['location']['distance_to_transport']
        location.is_main_road = scenario['location']['is_main_road']
        location.is_residential_area = scenario['location']['is_residential_area']
        location.safety_score = scenario['location']['safety_score']
        
        # Create review object
        reviews = ReviewMetrics()
        reviews.overall_rating = scenario['reviews']['overall_rating']
        reviews.num_reviews = scenario['reviews']['num_reviews']
        reviews.landlord_response_time = scenario['reviews']['landlord_response_time']
        reviews.maintenance_rating = scenario['reviews']['maintenance_rating']
        reviews.cleanliness_rating = scenario['reviews']['cleanliness_rating']
        reviews.value_rating = scenario['reviews']['value_rating']
        
        # Get price predictions
        prices = predict_monthly_prices(scenario['distance'], amenities, model, scaler, location, reviews)
        
        print(f"\nScenario: {scenario['name']}")
        print(f"Distance: {scenario['distance']}km")
        print(f"Amenities: {', '.join(prices['amenities'])}")
        print(f"Location Score: {prices['location_score']:.2f}")
        print(f"Review Score: {prices['review_score']:.2f}")
        print(f"Last Month Price: LKR {prices['last_month']:,.2f}")
        print(f"Current Price: LKR {prices['current']:,.2f}")
        print(f"Next Month Price: LKR {prices['next_month']:,.2f}")
        print(f"Next Year Estimate: LKR {prices['next_year_estimate']:,.2f}")
    
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

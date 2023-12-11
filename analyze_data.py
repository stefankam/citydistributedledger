import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# Load the data
data = pd.read_csv('./combined_public_transport_data.csv')  # Replace with your file path

# Encode the target variable if it's not already binary
le = LabelEncoder()
target_column = 'Method used to travel to workplace (12 categories)'
data[target_column] = le.fit_transform(data['Method used to travel to workplace (12 categories)'])

# Define the segments based on the actual values in the data
segments_conditions = {
    'Young_Urban_Professionals': (data['Age (6 categories)'].isin(['Aged 16 to 24 years','Aged 25 to 34 years'])) &
                                  (data['Occupation (current) (10 categories)'].str.contains('2. Professional occupations', '3. Associate professional and technical occupations')),

    'Families_with_Children': (data['Age (6 categories)'].isin(['Aged 35 to 49 years', 'Aged 50 to 64 years'])) &
     data['Occupation (current) (10 categories)'].str.contains('2. Professional occupations','1. Managers, directors and senior officials'),

    'Car_Owners': (data[target_column].isin(le.transform(['Driving a car or van','On foot']))) &
                  data['Distance travelled to work (5 categories)'].isin(['10km to less than 30km','30km and over','Works mainly from home','Not in employment or works mainly offshore, in no fixed place or outside the UK']),

    'Bike_Enthusiasts': (data[target_column].isin(le.transform(['Bicycle','On foot']))) &
                        data['Distance travelled to work (5 categories)'].isin(['less than 10km','Works mainly from home','Not in employment or works mainly offshore, in no fixed place or outside the UK']),

    'Eco_Conscious_Seniors': (data['Age (6 categories)'] == 'Aged 65 years and over') &
                             (data[target_column].isin(le.transform(['Train', 'Underground, metro, light rail, tram', 'Bus, minibus or coach']))),

    'Students': (data['Age (6 categories)'].isin(['Aged 16 to 24 years', 'Aged 25 to 34 years'])) &
                (data['Occupation (current) (10 categories)'].str.contains('9. Elementary occupations')),

    'Long_Distance_Travellers': (data['Distance travelled to work (5 categories)'] == '30km and over') &
                                (data[target_column].isin(le.transform(['Train','Driving a car or van'])))
}

# Define the bounding curve function
def bounding_curve(tokens):
    # Define your logic for how tokens impact adoption rate here
    # Redefine threshold and saturation tokens based on your range
    threshold_tokens = 0  # Adjust the threshold value as needed
    saturation_tokens = 10  # Adjust the saturation value as needed

    if isinstance(tokens, int) or isinstance(tokens, float):
        if tokens <= threshold_tokens:
            return 0.0
        else:
            return (tokens - threshold_tokens) / (saturation_tokens - threshold_tokens)

# Expand the behavior_model function
def behavior_model(segment_data):
    # Use relevant feature columns, ensure they are encoded if categorical
    X = segment_data[['Distance travelled to work (5 categories)', 'Age (6 categories)', 'Lower tier local authorities Code', 'Occupation (current) (10 categories)']].apply(LabelEncoder().fit_transform)  # Adjust columns as needed
    y = segment_data['Method used to travel to workplace (12 categories)']

  #  if len(y.unique()) <= 1 or len(segment_data) < 10:
  #      return y.iloc[0] if len(y) > 0 else None

    try:
        model = LogisticRegression()
        model.fit(X, y)
        predicted_probabilities = model.predict_proba(X)[:, 1]  # Adjust index if target class is different
        average_adoption_probability = np.mean(predicted_probabilities)
    except Exception as e:
        return None

    return average_adoption_probability

# Define the behavior_model function
def calculate_tokens_effect(segment_data, feature_columns, target_column):
    X = segment_data[feature_columns].apply(LabelEncoder().fit_transform)
    y = segment_data[target_column]

    try:
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model.fit(X, y)

        # Get the indices of the public transport classes actually present in the segment
        all_classes = le.classes_
        model_classes = model.classes_
        public_transport_classes = ['Train', 'Underground, metro, light rail, tram', 'Bus, minibus or coach']
        valid_classes = [cls for cls in public_transport_classes if cls in all_classes]
        valid_indices = [np.where(model_classes == le.transform([cls])[0])[0][0] for cls in valid_classes if le.transform([cls])[0] in model_classes]

        predicted_probabilities = model.predict_proba(X)
        if valid_indices:
            average_adoption_probability = np.mean(predicted_probabilities[:, valid_indices], axis=1)
        else:
            return None

    except Exception as e:
        print(f"Exception in model training: {e}")
        return None

    return np.mean(average_adoption_probability)


# Calculate the adoption rates for each segment
adoption_rates = {}
for segment_name, condition in segments_conditions.items():
    segment_data = data[condition]
    if not segment_data.empty:
        adoption_probability = behavior_model(segment_data)
        adoption_rate = bounding_curve(adoption_probability) if adoption_probability is not None else None
        adoption_rates[segment_name] = adoption_rate
    else:
        adoption_rates[segment_name] = None

print(adoption_rates)
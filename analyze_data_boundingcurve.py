import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('./combined_public_transport_data.csv')  # Replace with your file path

# Select the target variable and encode it
target_column = 'Method used to travel to workplace (12 categories)'
le = LabelEncoder()
data[target_column] = le.fit_transform(data[target_column])

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

# Define the bounding curve function
def calculate_tokens_effect(tokens):
    # Define your logic for how tokens impact adoption rate here
    # Redefine threshold and saturation tokens based on your range
    threshold_tokens = 0  # Adjust the threshold value as needed
    saturation_tokens = 10  # Adjust the saturation value as needed

    if isinstance(tokens, int) or isinstance(tokens, float):
        if tokens <= threshold_tokens:
            return 0.0
        else:
            return (tokens - threshold_tokens) / (saturation_tokens - threshold_tokens)

# Apply the tokens effect to the adoption probability
def apply_tokens_effect(adoption_probability, tokens_effect):
    if adoption_probability is not None:
        return np.minimum(adoption_probability + tokens_effect, 1)
    else:
        # Handle the case where adoption_probability is None
        return 0.0  # You can choose an appropriate default value

# Define a function to apply your rules
def assign_tokens(row):
    tokens = 0
    if row['Method used to travel to workplace (12 categories)'] == 'Bicycle':
        tokens += 3
    if row['Age (6 categories)'] == 'Aged 65 years and over' and row['Method used to travel to workplace (12 categories)'] in ['Bus, minibus or coach', 'Train']:
        tokens += 2
    if row['Age (6 categories)'] in ['Aged 16 to 24 years', 'Aged 25 to 34 years'] and row['Method used to travel to workplace (12 categories)'] in ['public_transport_classes']:
        tokens += 2
    if row['Distance travelled to work (5 categories)'] == '30km and over' and row['Method used to travel to workplace (12 categories)'] == 'Train':
        tokens += 4
    return tokens
# Apply the function to each row to calculate the predicted tokens
data['Predicted_Tokens'] = data.apply(assign_tokens, axis=1)

# Calculate the adoption rates for each segment
adoption_rates = {
    'Linear': {},
}

feature_columns = ['Distance travelled to work (5 categories)', 'Age (6 categories)', 'Lower tier local authorities Code', 'Occupation (current) (10 categories)']
for segment_name, condition in segments_conditions.items():
    segment_data = data[condition]
    if not segment_data.empty:
        adoption_probability = behavior_model(segment_data)
        print('segment_name',segment_name)
        print('adoption_probability: ',adoption_probability)
        # Apply the bounding curves to the adoption probability
        #segment_data['Adoption_Rate_Linear'] = linear_bounding_curve(adoption_probability)
        #segment_data['Adoption_Rate_Diminishing'] = diminishing_returns_bounding_curve(adoption_probability)
        # Calculate the mean adoption rate for the segment
        tokens_effect = calculate_tokens_effect(adoption_probability)
        print('tokens_effect: ', tokens_effect)
        # Apply the tokens effect to the adoption rate for the segment
        adoption_rates['Linear'][segment_name] = apply_tokens_effect(adoption_probability, tokens_effect)
    else:
        adoption_rates['Linear'][segment_name] = None

print(adoption_rates)

import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px  # Import Plotly Express for plotting

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


# Define a function to apply your rules
def assign_tokens(row):
    tokens = 0
    if row['Method used to travel to workplace (12 categories)'] == 'Bicycle':
        tokens += 3
    if row['Age (6 categories)'] == 'Aged 65 years and over' and row['Method used to travel to workplace (12 categories)'] in ['public_transport_classes']:
        tokens += 2
    if row['Age (6 categories)'] in ['Aged 16 to 24 years','Aged 25 to 34 years'] and row['Method used to travel to workplace (12 categories)'] in ['public_transport_classes']:
        tokens += 2
    if row['Age (6 categories)'] in ['Aged 35 to 49 years','Aged 50 to 64 years'] and row['Method used to travel to workplace (12 categories)'] in ['public_transport_classes']:
        tokens += 2
    if row['Distance travelled to work (5 categories)'] == '30km and over' and row['Method used to travel to workplace (12 categories)'] in ['public_transport_classes']:
        tokens += 4
    return tokens

# Calculate the tokens effect based on your specific criteria
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
    else:
        # Handle cases where tokens is not a single number, e.g., a Series or DataFrame column
        result = []
        for token in tokens:
            if token <= threshold_tokens:
                result.append(0.0)
            else:
                return (tokens - threshold_tokens) / (saturation_tokens - threshold_tokens)
        return np.array(result)

# Apply the tokens effect to the adoption probability
def apply_tokens_effect(adoption_probability, tokens_effect):
    if adoption_probability is not None:
        return np.minimum(adoption_probability + tokens_effect, 1)
    else:
        # Handle the case where adoption_probability is None
        return 0.0  # You can choose an appropriate default value


# Initialize a list to store adoption rates for different tokens
adoption_rates = []

# Apply the function to each row to calculate the predicted tokens
data['Predicted_Tokens'] = data.apply(assign_tokens, axis=1)

# Create a dictionary to store adoption rates for each segment
segment_adoption_tables = {}

feature_columns = ['Distance travelled to work (5 categories)', 'Age (6 categories)',
                   'Lower tier local authorities Code', 'Occupation (current) (10 categories)']


# Calculate adoption rates for different token values
for token_value in range(0, 11):  # Assuming tokens range from 0 to 10
    segment_data = data.copy()  # Create a copy of the data for each token value
    # Modify it to set the same 'token_value' for all rows in 'segment_data':
    segment_data['Predicted_Tokens'] = np.full(len(segment_data), token_value)

    adoption_probability = behavior_model(segment_data)
    print('adoption_probability: ', adoption_probability)
    if adoption_probability is not None:
        # Calculate the tokens effect
        tokens_effect = calculate_tokens_effect(segment_data['Predicted_Tokens'])
        print('tokens_effect: ', tokens_effect)
        # Apply the tokens effect to the adoption rate
        adoption_rate_adjusted = apply_tokens_effect(adoption_probability, tokens_effect)
        print('adoption_rate_adjusted: ', adoption_rate_adjusted)
        adoption_rates.append((token_value, adoption_rate_adjusted))

        # Calculate adoption rates for each segment
        for segment_name, condition in segments_conditions.items():
            segment_data_segment = segment_data[condition]
            print("segment_data_segment: ", segment_data_segment)
            if not segment_data_segment.empty:
                adoption_probability_segment = behavior_model(segment_data_segment)
                print("adoption_probability_segment: ", adoption_probability_segment)
                # Calculate the tokens effect for the segment
                tokens_effect_segment = calculate_tokens_effect(segment_data_segment['Predicted_Tokens'])
                print("tokens_effect_segment: ", tokens_effect_segment)
                # Apply the tokens effect to the adoption rate for the segment
                adoption_rate_adjusted_segment = apply_tokens_effect(adoption_probability_segment, tokens_effect_segment)
                segment_adoption_tables.setdefault(segment_name, []).append(
                    {'Token_Value': token_value, 'Adoption_Rate': adoption_rate_adjusted_segment})


# Initialize an empty dictionary for data
data = {'Segment': [], 'Token Value': [], 'Adoption Rate': []}
# Populate the data dictionary with your data
for segment_name, rates in segment_adoption_tables.items():
    for rate_data in rates:
        data['Segment'].append(segment_name)
        data['Token Value'].append(rate_data['Token_Value'])
        data['Adoption Rate'].append(rate_data['Adoption_Rate'])

# Create a DataFrame from the data
table_df = pd.DataFrame(data)

# Print the table
print(table_df)
table_df.to_csv('table_data.csv', index=False)

# Create a plot (replace this with your code)
fig = px.line(table_df, x='Token Value', y='Adoption Rate', color='Segment',
              title='Adoption Rate vs Token Value for Different Segments')

# Save the plot as an HTML file
fig.write_html('plot.html')
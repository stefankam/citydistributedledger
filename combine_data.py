import pandas as pd

# Load datasets
distance_data = pd.read_csv('./data/distance_travel_age.csv')
method_region_data = pd.read_csv('./data/method_travel_region.csv')
method_age_data = pd.read_csv('./data/method_travel_age.csv')
method_occupation_data = pd.read_csv('./data/method_travel_occupation.csv')

# Assuming 'Common_Column' is the common column across datasets
# Replace 'Common_Column' with the actual common column name
# Adjust the 'how' parameter based on your merging criteria (inner, outer, left, right)
combined_data = pd.merge(distance_data, method_region_data, on='Lower tier local authorities Code', how='inner')
# combined_data = pd.merge(combined_data, method_age_data, on='Lower tier local authorities', how='inner', suffixes=('', '_age'))
combined_data = pd.merge(combined_data, method_occupation_data, on='Lower tier local authorities Code', how='inner', suffixes=('', '_occ'))

# Preprocess the combined data (handle missing values, encode categorical variables, etc.)

# Save the combined dataframe to a new CSV file
combined_data.to_csv('combined_public_transport_data.csv', index=False)

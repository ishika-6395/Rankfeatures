import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate Dataset
num_entries = 200000
user_ids = np.random.randint(1000, 9999, num_entries)
feature_types = ['Hardware', 'Software']
hardware_requests = ['Camera', 'Battery', 'RAM', 'Screen', 'Processor']
software_requests = ['OS', 'Application suite', 'Security', 'UI/UX']
feature_type = np.random.choice(feature_types, num_entries)
hardware_request = np.random.choice(hardware_requests, num_entries)
software_request = np.random.choice(software_requests, num_entries)

data = {
    'User ID': user_ids,
    'Feature Type': feature_type,
    'Hardware Request': np.where(feature_type == 'Hardware', hardware_request, None),
    'Software Request': np.where(feature_type == 'Software', software_request, None)
}

df = pd.DataFrame(data)
df.to_csv('feature_requests.csv', index=False)

# Step 2: Load and Clean Data
df = pd.read_csv('feature_requests.csv')
print("Loaded Data:")
print(df.head())

# Handle missing values by filling them with a placeholder
df['Hardware Request'].fillna('None', inplace=True)
df['Software Request'].fillna('None', inplace=True)
print("Cleaned Data:")
print(df.head())

# Step 3: Analyze Data
hardware_counts = df[df['Feature Type'] == 'Hardware']['Hardware Request'].value_counts()
software_counts = df[df['Feature Type'] == 'Software']['Software Request'].value_counts()
print("Hardware Counts:")
print(hardware_counts)
print("Software Counts:")
print(software_counts)

# Step 4: Rank Features
combined_counts = pd.concat([hardware_counts, software_counts])
ranked_features = combined_counts.sort_values(ascending=False)
print("Ranked Features:")
print(ranked_features)

# Step 5: Display Rankings
plt.figure(figsize=(10, 6))
sns.barplot(x=ranked_features.values, y=ranked_features.index)
plt.xlabel('Number of Requests')
plt.ylabel('Feature')
plt.title('Feature Request Rankings')
plt.show()

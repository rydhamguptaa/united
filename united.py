# MySQL connection setup
conn = mysql.connector.connect(
    host="localhost",   # or the host where your MySQL server is running
    user="your_username",
    password="your_password",
    database="your_database_name"
)

# Define a function to load data from MySQL into pandas
def load_data_from_sql(query, conn):
    return pd.read_sql(query, conn)
# Load the data from respective tables
calls = load_data_from_sql("SELECT * FROM calls", conn)
customers = load_data_from_sql("SELECT * FROM customers", conn)
reason = load_data_from_sql("SELECT * FROM reason", conn)
sentiment_statistics = load_data_from_sql("SELECT * FROM sentiment_statistics", conn)

# Preview the data
calls.head(), customers.head(), reason.head(), sentiment_statistics.head()
# Convert datetime columns to pandas datetime objects
calls['call_start_datetime'] = pd.to_datetime(calls['call_start_datetime'])
calls['agent_assigned_datetime'] = pd.to_datetime(calls['agent_assigned_datetime'])
calls['call_end_datetime'] = pd.to_datetime(calls['call_end_datetime'])

# Calculate AHT and AST
calls['AHT'] = (calls['call_end_datetime'] - calls['agent_assigned_datetime']).dt.total_seconds()
calls['AST'] = (calls['agent_assigned_datetime'] - calls['call_start_datetime']).dt.total_seconds()

# Drop any rows with missing or invalid values
calls.dropna(subset=['AHT', 'AST'], inplace=True)
# Merge customers and calls on customer_id
merged_data = pd.merge(calls, customers, on='customer_id', how='left')

# Merge reason on call_id
merged_data = pd.merge(merged_data, reason, on='call_id', how='left')

# Merge sentiment statistics on call_id
merged_data = pd.merge(merged_data, sentiment_statistics, on=['call_id', 'agent_id'], how='left')

# Preview the merged dataset
merged_data.head()
# AHT and AST summary statistics
print("AHT Summary:")
print(merged_data['AHT'].describe())

print("AST Summary:")
print(merged_data['AST'].describe())

# Visualize AHT and AST distributions
sns.histplot(merged_data['AHT'], kde=True, bins=30).set(title='AHT Distribution')
plt.show()

sns.histplot(merged_data['AST'], kde=True, bins=30).set(title='AST Distribution')
plt.show()
# Group data by primary call reason and compute mean AHT and AST
reason_aht_ast = merged_data.groupby('primary_call_reason').agg(
    mean_aht=('AHT', 'mean'),
    mean_ast=('AST', 'mean'),
    count=('call_id', 'count')
).reset_index()

# Sort by AHT
reason_aht_ast = reason_aht_ast.sort_values(by='mean_aht', ascending=False)

# Display top call reasons by AHT
print(reason_aht_ast.head(10))

# Plotting
plt.figure(figsize=(12,6))
sns.barplot(x='mean_aht', y='primary_call_reason', data=reason_aht_ast.head(10), palette='viridis').set(title='Top Call Reasons by AHT')
plt.show()


# Group by agent and calculate average AHT and AST
agent_performance = merged_data.groupby('agent_id').agg(
    mean_aht=('AHT', 'mean'),
    mean_ast=('AST', 'mean'),
    silence_avg=('silence_percent_average', 'mean'),
    call_count=('call_id', 'count')
).reset_index()

# Sort by AHT to find agents with the highest AHT
agent_performance = agent_performance.sort_values(by='mean_aht', ascending=False)
print(agent_performance.head())

# Visualizing agent performance
plt.figure(figsize=(12,6))
sns.barplot(x='mean_aht', y='agent_id', data=agent_performance.head(10), palette='coolwarm').set(title='Top Agents by AHT')
plt.show()
# Frequent call reasons with high AHT
high_aht_reasons = reason_aht_ast[reason_aht_ast['count'] > 50].sort_values(by='mean_aht', ascending=False)
print(high_aht_reasons.head())
# Identify calls with self-solvable issues
self_service_calls = merged_data[merged_data['primary_call_reason'].isin(['Flight Status', 'Baggage Tracking', 'Booking Inquiry'])]

# Calculate potential savings in AHT if these calls were automated
potential_aht_reduction = self_service_calls['AHT'].sum()
print(f"Potential reduction in AHT if self-solvable issues are automated: {potential_aht_reduction} seconds")
# Extract features from datetime columns
merged_data['call_start_hour'] = merged_data['call_start_datetime'].dt.hour

# Convert categorical features to numeric using one-hot encoding
merged_data_encoded = pd.get_dummies(merged_data[['customer_id', 'agent_id', 'elite_level_code', 'call_start_hour']])

# Define feature set and target variable
X = merged_data_encoded
y = merged_data['primary_call_reason']
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Export predictions (for test.csv)
test_predictions = pd.DataFrame({'call_id': X_test.index, 'primary_call_reason': y_pred})
test_predictions.to_csv('test_predictions.csv', index=False)



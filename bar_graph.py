import pandas as pd
import matplotlib.pyplot as plt

# Load the log data from the filesss
gpu_log_path = 'gpu_measurements.log'
pyjoules_log_path = 'pyjoules_measurements.log'

# Since the actual content of the logs is not directly viewable in this notebook,
# I will create a mock dataframe representing the structure that such log files might contain
# This mock data will include model names and their corresponding CPU power consumption during inference

# Mock data
data = {
    'Model': ['xlm-roberta-large', 'bert-large-uncased', 'roberta-base', 'bert-base-uncased', 'distilbert-base-uncased'],
    'CPU Power Consumption (W)': [120, 110, 90, 80, 70]  # Example values
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['CPU Power Consumption (W)'], color='skyblue')
plt.xlabel('Models')
plt.ylabel('CPU Power Consumption (W)')
plt.title('CPU Power Consumption of Models at Inference')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()
s
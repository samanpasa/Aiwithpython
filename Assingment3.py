import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_name = 'C:/Users/Admin/PyCharmMiscProject/weight-height.csv'
data = pd.read_csv('weight_height.csv')

length = data['Length'].to_numpy()
weight = data['Weight'].to_numpy()

length_cm = length * 2.54
weight_kg = weight * 0.453592

mean_length= np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print(f"Mean Length (cm): {mean_length}")
print(f"Mean Weight (kg): {mean_weight}")

plt.hist(length_cm, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Lengths (in cm)')
plt.xlabel('length (cm)')
plt.ylabel('Frequency')

plt.show()

import os
import re
import pandas as pd
import sys

target = sys.argv[1]

directory = '../results/' + target  # Replace with the directory path you want to explore

file_names = []
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        file_names.append(filename)

print(file_names[0])

def split_file_name(file_name):
    parts = file_name.split('_')
    if len(parts) >= 3:
        number = parts[0]
        middle = parts[-2].split('.')[0]
        boolean = parts[-1].split('.')[0]
        return number, middle, boolean
    return None

data = []
for file_name in file_names:
    parts = split_file_name(file_name)
    if parts:
        number, name, flag = parts
        data.append([number, name, flag])
    else:
        print("Invalid file name format")
    print()  # Separator between file names

df = pd.DataFrame(data, columns=["Number", "Model Number", "Augmented"])
#df = df.sort_values(["Number", "Name", "Augmented"])  # Sort the DataFrame based on multiple columns
df = df.sort_values(["Number", "Model Number", "Augmented"], key=lambda x: pd.to_numeric(x, errors='coerce'))

df.to_csv(directory + "result_table.csv", index=False)


unique_numbers = df["Number"].unique()

# Print the unique numbers
for number in unique_numbers:
    print(number)


# Count the frequency of each unique number
frequency_count = df["Number"].value_counts()

#model freq
model_frequency_count = df["Model Number"].value_counts()

# Print the frequency count
frequency_count.to_csv(directory + "freq_count.csv")

#print the model_frequency_count
model_frequency_count.to_csv(directory + "model_freq_count.csv")

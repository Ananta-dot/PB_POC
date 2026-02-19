
import pickle

# Replace 'your_file.pkl' with the path to your pickle file
file_name = 'misr_elites_1.pkl'

try:
    # Open the file in binary read mode ('rb')
    with open(file_name, 'rb') as f:
        # Load the data from the file
        data = pickle.load(f)
        
        # Print the loaded data
        print("Successfully loaded data from pickle file:")
        print(data)

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
except EOFError:
    print(f"Error: The file '{file_name}' is empty or corrupted.")
except Exception as e:
    print(f"An error occurred: {e}")
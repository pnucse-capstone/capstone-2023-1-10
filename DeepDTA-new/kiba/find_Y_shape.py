import pickle

# Load data from the Pickle file

#Y = pickle.load(open("C:/Users/admin/Desktop/DeepDTA/DeepDTA-new/kiba/Y", "rb"))
Y = pickle.load(open("C:/Users/admin/Desktop/DeepDTA/DeepDTA-new/davis/Y", "rb"), encoding='latin1')
lowest_value = float('inf')
for row in Y:
    for item in row:
        if item < lowest_value:
            lowest_value = item

    
print("The lowest value is:", lowest_value)
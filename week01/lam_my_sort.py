# Get input from the user
input_str = input("Give a list of integers separated by space: ")

# Split the input string into a list of integers
input_list = []
for i in input_str.split():
    input_list.append(int(i))

# Sort the list of integers
sorted_list = sorted(input_list)

# Print the sorted list
print("Given numbers sorted:", sorted_list)






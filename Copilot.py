import json

# Open the file in read mode
with open(r'D:\python\Distance_measurement\output.txt', 'r') as f:
    # Read the file content
    content = f.read()

# Parse the JSON data
# data = json.loads(content)

# # Extract the parameters
# intrinsic_parameters = data['intrinsic_parameters']
# distortion_parameters = data['distortion_parameters']

# # Print the parameters
# print('Intrinsic Parameters: ', intrinsic_parameters)
# print('Distortion Parameters: ', distortion_parameters)
print(content)
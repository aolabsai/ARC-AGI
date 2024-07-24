

import numpy as np
import json
import os
import random

# assumes an available local installation of ao_core; refer to https://github.com/aolabsai/ao_core?tab=readme-ov-file#installing-ao_core
import ao_core as ao
neurons = 20 #its a global variable for number of neurons 
description = "ARC Agent"      #    MNIST is in grayscale, which we downscaled to B&W for the simple 28x28 neuron count -- 788 = 28x28 + 4
arch_i = [neurons*neurons * 4]               # note that the 784 I neurons are in 1 input channel; MNIST is like a single channel clam, so it's limitations are obvious from the prespective of our approach, more on this here: 
arch_z = [neurons*neurons * 4]                   # 4 neurons in 1 channel as 4 binary digits encodes up to integer 16, and only 10 (0-9) are needed for MNIST
arch_c = []
connector_function = "full_conn"
# connector_parameters = [1200, 1200, 1200, 1200]
# so all Q neurons are connected randomly to 1200 I and 1200 neighbor Q
# and all Z neurons are connected randomly to 784 Q and 4 (or all) neighbor Z

# To maintain compatability with our API, do not change the variable name "Arch" or the constructor class "ao.Arch" in the line below (the API is pre-loaded with a version of the Arch class in this repo's main branch, hence "ao.Arch")
arcArch = ao.Arch(arch_i, arch_z, arch_c, connector_function, description)
arcAgent = ao.Agent( arcArch )


#Change this path accordingly
path = "E:/aolabs/aolabs2/ARC-AGI/data/training/"  


##padding function

def pad_ARC(arr, pad_value=10, final_size=(neurons, neurons)):
    # Get the current size of the array
    current_size = arr.shape
    
    # Initialize the padding list
    padding = []
    
    # Calculate the amount of padding needed for each dimension
    for i in range(len(final_size)):
        total_pad = max(0, final_size[i] - current_size[i])
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        padding.append((pad_before, pad_after))
    
    # Pad the array
    padded_arr = np.pad(arr, padding, mode='constant', constant_values=pad_value)
    
    return padded_arr



color_to_binary = [
    '0000', # black
    format(1, '04b'), # blue
    format(2, '04b'), # red
    format(3, '04b'), # green
    format(4, '04b'), # yellow
    format(5, '04b'), # grey
    format(6, '04b'), # pink
    format(7, '04b'), # orange
    format(8, '04b'), # l blue
    format(9, '04b'), # maroon
    '1010', # void / null   #asigning it 1010, because its 10(decimal) in binary.
]


def ARC_to_binary( input_padded):

    input_flat = input_padded.flatten()
    inn_stringvec = ""
    for p in input_flat:
        inn_stringvec += color_to_binary[p]  #this line adds 4 bits 
    
    inn_narray = np.asarray(list(inn_stringvec), dtype=int)

    return inn_narray


#function to convert binary color to integer color
def binary_to_color(binary_color):
    decimal = 0
    for digit in binary_color:
        decimal = decimal*2 + int(digit)
    return decimal   

def binary_to_ARC(binary_input, original_shape=(neurons,neurons)):
    # Initialize an empty list to hold the chunks of 4 bits
    chunks = []
    
    # Split the binary input into chunks of 4 bits
    for i in range(0, len(binary_input), 4):
        chunk = ''.join(map(str, binary_input[i:i+4]))
        chunks.append(chunk)
    
    # Initialize an empty list to hold the color indices
    color_indices = []
    
    # Convert binary chunks back to their original color indices
    for chunk in chunks:
        color_index = binary_to_color(chunk)
        color_indices.append(color_index)
    
    # Convert the list of color indices to a numpy array and reshape it to the original shape
    output_array = np.array(color_indices).reshape(original_shape)
    
    return output_array


def depad_ARC(arr, pad_value=10):
    # Get the current size of the array
    current_size = arr.shape
    
    # Initialize the indices for slicing
    slice_indices = []
    
    # Calculate the indices for each dimension
    for i in range(len(current_size)):
        start = 0
        end = current_size[i]
        
        # Find the first non-pad_value index from the beginning
        while start < end:
            if np.all(arr.take(indices=start, axis=i) == pad_value):
                start += 1
            else:
                break
        
        # Find the first non-pad_value index from the end
        while end > start:
            if np.all(arr.take(indices=end-1, axis=i) == pad_value):
                end -= 1
            else:
                break
        
        slice_indices.append(slice(start, end))
    
    # Slice the array to remove padding
    depadded_arr = arr[tuple(slice_indices)]
    
    return depadded_arr



files = os.listdir(path)

# Randomly sample 10 files from the list
tasks = random.sample(files, 10)  # Picking 10 random files from the training folder

# Process each sampled file
for task in tasks:
    # Construct the full path for the current file
    task_path = path + task

    # Open the JSON file and load its content
    with open(task_path) as task_open:
        task_data = json.load(task_open)
    
    # Process each training example in the file
    for pair in task_data['train']:
        inp = np.asarray(pair['input'])  # Convert input data to NumPy array
        # Pad the input array
        inp_padded = pad_ARC(inp)
        # Convert the padded input array to binary format
        inp_binary = ARC_to_binary(inp_padded)

        onp = np.asarray(pair['output'])  # Convert output data to NumPy array
        # Pad the output array
        onp_padded = pad_ARC(onp)
        # Convert the padded output array to binary format
        onp_binary = ARC_to_binary(onp_padded)

        # Reset the state of the arcAgent
        arcAgent.reset_state()
        # Train the arcAgent with the input binary data and the label
        arcAgent.next_state(inp_binary, LABEL=onp_binary, unsequenced=True)  # Training with label on

    # Get the number of test examples
    test_len = len(task_data['test'])

    output_arr = []  # List to store output arrays for each test example
    
    # Process each test example in the file
    for pair in task_data['test']:
        inp = np.asarray(pair['input'])  # Convert input data to NumPy array
        # Pad the input array
        inp_padded = pad_ARC(inp)
        # Convert the padded input array to binary format
        inp_binary = ARC_to_binary(inp_padded)

        onp = np.asarray(pair['output'])  # Convert output data to NumPy array
        # Pad the output array
        onp_padded = pad_ARC(onp)
        # Convert the padded output array to binary format
        onp_binary = ARC_to_binary(onp_padded)

        # Run the arcAgent multiple times for prediction
        for run in range(10):
            # Get the next state of the arcAgent
            arcAgent.next_state(inp_binary)
            z_index = arcAgent.arch.Z__flat  # Get the current index from the architecture
            s = arcAgent.state - 1  # Get the state index
            response = arcAgent.story[s, z_index]  # Get the response from the story

        # Convert the binary response back to array format
        arr_op_pad = binary_to_ARC(response)
        # Remove padding from the predicted output array
        arr_op = depad_ARC(arr_op_pad)
        output_arr.append(arr_op)  # Store the predicted output array

        # Get the shapes of the actual and predicted output arrays
        actual_op_size = onp.shape
        pred_op_size = arr_op.shape

        # Print the predicted and actual output arrays
        print("Predicted output: ")
        print(arr_op)
        print("Actual output: ")
        print(onp)

        # Compare the shapes of the actual and predicted output arrays
        if (actual_op_size[0] != pred_op_size[0]) or (actual_op_size[1] != pred_op_size[1]):
            print('Wrong solution because of incorrect size')
        else:
            # Compare each element in the actual and predicted output arrays
            for i in range(actual_op_size[0]):
                for j in range(actual_op_size[1]):
                    if (onp[i][j] != arr_op[i][j]):
                        print("Wrong output")
                        break
                else:
                    continue
                break
            else:
                print("Correct output")


    

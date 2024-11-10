import numpy as np
import json
import os
import random

# assumes an available local installation of ao_core; refer to https://github.com/aolabsai/ao_core?tab=readme-ov-file#installing-ao_core
import ao_core as ao
import ao_arch as ar



neurons_x = 30  # Number of neurons in the x direction (global variable)
neurons_y = 30  # Number of neurons in the y direction

def setup_agent():
    # Description of the agent
    description = "ARC-AGI agent"  
    
    # Initialize the input and output architecture with 4 neurons per channel
    # Input architecture (4 neuron per channel for encoding colors in binary) 
    # arch_i = [4 for _ in range(neurons_x * neurons_y)]  
    arch_i = [4 for _ in range(9)]  
    # Output architecture 
    arch_z = [4] 
    arch_c = []  

    # Function for connecting neurons
    # connector_function = "nearest_neighbour_conn"  
    # # wether want Z to I connection or not. If not specified, by default it's False. 
    # Z2I_connections = False 
    # # ax, dg, neurons_x, neurons_y and Z2I connection (True or default False)
    # connector_parameters = [5, 5, neurons_x, neurons_y, Z2I_connections]  
    
    # connector_function = "rand_conn"
    # connector_parameters = [3600*.3, 3600*.3, 3600*.3, 3600*.3]

    connector_function = "full_conn"
    connector_parameters = []

    # Create the architecture using the Arch class from the ao_arch library
    arcArch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)
    
    
    arcAgent = ao.Agent( arcArch, save_meta=False, _steps=100000 )
    return arcAgent



def rectangle_points(row, col, x, y, size=(30, 30)):
    # Boundaries for the grid
    max_row, max_col = size

    # Calculating the boundaries considering the size limits
    top_row = max(0, row - y)
    left_col = max(0, col - x)

    bottom_row = min(max_row - 1, row + y)
    right_col = min(max_col - 1, col + x)

  
    # Collecting all coordinates inside the rectangle
    points = []
    for r in range(top_row, bottom_row + 1):
        for c in range(left_col, right_col + 1):
            points.append((r, c))
    
    return points

##padding function

def pad_ARC(arr, pad_value=10, final_size=(neurons_y, neurons_x)):
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

def binary_to_ARC(binary_input, original_shape=(neurons_y,neurons_x)):
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


def ARC_main(arcAgent, tasks):
    Data = []
    
    arcAgent.reset_state()
    arcAgent.reset_state(training=True)
    arcAgent.reset_state(training=True)
    # arcAgent.reset_state(training=True)
    # arcAgent.reset_state()
    # arcAgent.reset_state(training=True)
    # arcAgent.reset_state()
    # arcAgent.reset_state(training=True)

    for task in tasks:
        print('Training going on for task..', task)
        # Construct the full path for the current file
        path = "data/training/" 
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
            # arcAgent.next_state(inp_binary, LABEL=onp_binary)  # Training with label on
            r = 0
            c = 0
            iii=0
            focus_input_batch = np.zeros([900, 9*4])
            LABEL_batch = np.zeros([900, 4])
            for i in np.arange(0, 3600, 4):
                print("TRAINING: "+str(i)+" - "+ str(r)+", "+str(c)) 
                ii = 0
                focus_inp_padded = np.zeros(9, dtype=int)
                points = rectangle_points(r, c, 1, 1, (30,30))
                for p in points:
                    focus_inp_padded[ii] = inp_padded[p[0], p[1]]
                    ii += 1
                focus_input = ARC_to_binary(focus_inp_padded)
                focus_input_batch[iii, :] = focus_input 
                # focus_input_batch[iii] = focus_input
                LABEL=onp_binary[i:i+4]
                # LABEL_batch[iii] = LABEL
                LABEL_batch[iii, :] = LABEL
                print("SHAPE LABEL "+ str(LABEL_batch.shape))
                print("SHAPE INPUT "+ str(focus_input_batch.shape))
                # arcAgent.next_state(focus_input, LABEL=onp_binary[i:i+4])  # Training with label on
                iii += 1
                c += 1
                if c == 30:
                    c = 0
                    r += 1
                    if r == 30:
                        r = 0
            arcAgent.next_state_batch(focus_input_batch, LABEL_batch, unsequenced=True)

        # Get the number of test examples
        test_len = len(task_data['test'])
        arcAgent.reset_state()   # <------

        
        file_data = []
        # Process each test example in the file
        for pair in task_data['test']:
            test_data = []
            inp = np.asarray(pair['input'])  # Convert input data to NumPy array
            # Pad the input array
            inp_padded = pad_ARC(inp)
            # Convert the padded input array to binary format
            inp_binary = ARC_to_binary(inp_padded)
            
            # arcAgent.story[arcAgent.state-1, arcAgent.arch.Q__flat] = inp_binary   # <------
            
            onp = np.asarray(pair['output'])  # Convert output data to NumPy array
            # Pad the output array
            onp_padded = pad_ARC(onp)
            # Convert the padded output array to binary format
            onp_binary = ARC_to_binary(onp_padded)

            # Run the arcAgent multiple times for prediction
            for run in range(3):
                # Get the next state of the arcAgent
                # arcAgent.next_state(inp_binary)
                response = np.zeros(3600, dtype=int)
                r = 0
                c = 0
                for i in np.arange(0, 3600, 4):
                    print("TESTINGGGGG: "+ str(r)+", "+str(c)) 
                    ii = 0
                    focus_inp_padded = np.zeros(9, dtype=int)
                    points = rectangle_points(r, c, 1, 1, (30,30))
                    for p in points:
                        focus_inp_padded[ii] = inp_padded[p[0], p[1]]
                        ii += 1
                    focus_input = ARC_to_binary(focus_inp_padded)
                    response[i:i+4] = arcAgent.next_state(focus_input, DD=False)  # Training with label on
                    c += 1
                    if c == 30:
                        c = 0
                        r += 1
                        if r == 30:
                            r = 0
                print(response.shape)
                print(response)
                # z_index = arcAgent.arch.Z__flat  # Get the current index from the architecture
                # q_index = arcAgent.arch.Q__flat
                s = arcAgent.state - 1  # Get the state index

                # print('S:', s)
                # print(z_index[166])
                # response = arcAgent.story[s, z_index]  # Get the response from the story
                # response = np.asarray(response)
                # response_q = arcAgent.story[s, q_index]
                response_q = np.zeros(4*30*30, dtype=int)
                arr_op_pad = binary_to_ARC(response)
                q_arr_op_pad = binary_to_ARC(response_q)
                arr_op = depad_ARC(arr_op_pad).tolist()
                q_arr_op = depad_ARC(q_arr_op_pad).tolist()
                pr = [arr_op,q_arr_op]
                test_data.append(pr) 

            file_data.append(test_data)
            arcAgent.reset_state()   # <------

        Data.append(file_data)   
        print('Training Done for Task ', task) 
        

    return Data   


# Data[task index][test case index][state index][Z or Q state index]

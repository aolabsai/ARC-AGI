import ao_arch as ar

neurons_x = 30  # Number of neurons in the x direction (global variable)
neurons_y = 30  # Number of neurons in the y direction
description = "ARC-AGI agent"  # Description of the agent

# Initialize the input and output architecture with 4 neurons per channel
arch_i = [4 for x in range(neurons_x * neurons_y)]  # Input architecture (4 neuron per channel for encoding colors in binary) 
arch_z = [4 for x in range(neurons_x * neurons_y)]  # Output architecture 
arch_c = [1 for x in range(neurons_x * neurons_y)]
connector_function = "nearest_neighbour_conn"  # Function for connecting neurons


Z2I_connections = True #wether want Z to I connection or not. If not specified, by default it's False. 
connector_parameters = [4, 4, neurons_x, neurons_y, Z2I_connections]  #ax, dg, neurons_x, neurons_y and Z2I connection (True or default False)

# Create the architecture using the Arch class from the ao_arch library
arcArch = ar.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)


## some code to connect each C channel/neuron to corresponding Q and Z channels
channel_num = 0
for channel in arcArch.Q:
    for neuron in channel:
        arcArch.datamatrix[3, neuron] = arcArch.C[channel_num]
        channel_num += 1

channel_num = 0
for channel in arcArch.Z:
    for neuron in channel:
        arcArch.datamatrix[3, neuron] = arcArch.C[channel_num]
        channel_num += 1


##### in ARC application,
## some code to trigger each C channel/neuron if the corresponding Z channel gets the right answer

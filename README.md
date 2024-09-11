# ARC-AGI aolabs agent
**Author & maintainer:** [kushagra7777](https://github.com/Kushagra7777), [kushagra@aolabs.ai].

**Description:** ARC is a general AI benchmark and here at AO Labs we are trying to solve it with Weightless Neural Network (WNN) architecture which is a different approach than deep learning. 

## Introduction

The ARC challenge is to create an AI that can solve different kinds of visual puzzles. These puzzles require the AI to think abstractly and generalize its understanding based on limited training examples. The goal is to build an AI that can figure out and apply logical rules to solve new puzzles it hasn't seen before.


## Local installation steps
1. **Clone the repository**

   ```bash
   git clone https://github.com/aolabsai/ARC-AGI
   cd ARC-AGI
   ```
2. **Create and activate a Python virtual environment**

   ```bash
   python -m venv myenv
   \myenv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Activate the Backend**

   ```bash
   python app.py
   ```
5. Go to apps folder and open `testing_interface.html` file in the browser


### Docker Installation

1) Generate a GitHub Personal Access Token to ao_core    
    Go to https://github.com/settings/tokens?type=beta

2) Clone this repo and create a `.env` file in your local clone where you'll add the PAT as follows:
    `ao_github_PAT=token_goes_here`
    No spaces! See `.env_example`.

3) In a Git Bash terminal, build and run the Dockerfile with these commands:
```shell
export DOCKER_BUILDKIT=1

docker build --secret id=env,src=.env -t "ao_app" .

docker run -p 5000:5000 "ao_app"
```
You're done! Access the app at `http://localhost:8501/` in your browser. It will automatically led you to ARC app interface. 



## Usage

The agent architecture is predefined in the `arch_ARC.py` file. It is designed to train on a training sample for a given task and then display the predicted output on a web page. 


### Task Workflow Overview

Agents have 3 layers here, an input layer, state layer, and output layer.

1. **Padding Training Puzzles**  
   - All training puzzles (both input and output) are padded to a size of 30x30.
   - A new null color is defined specifically for padding purposes.

2. **Binary Conversion**  
   - The padded arrays are converted into binary arrays.
   - Each color in the puzzle is represented by a four-digit binary code.
   - These binary codes are stored in new binary array variables.

3. **Training the Agent**  
   - The agent is trained using these binary arrays.
   - The agent operates using a nearest-neighbor connection strategy, where only neighboring neurons are connected.
   - The number of neighbors is defined by the `connector_parameters` variable in the code.

4. **Testing the Agent**  
   - For testing, the fresh test input is first padded in the same way as the training puzzles.
   - The padded test input is then converted into a binary format.
   - The binary test input is fed into the agent to obtain the solution in binary form.

5. **Post-Processing Output**  
   - The agent's output (in binary) is converted back to its original format (de-binarization).
   - The result is depadded in the same manner as the original padding step.
   - The final output is then displayed on the webpage.


## File Structure

The `arch_ARC.py` file defines the structure and architecture of the agent used in the project. Training and testing puzzles, provided by [ARC-AGI](https://github.com/fchollet/ARC-AGI), are stored in the `Data` folder in JSON format. The `apps` folder contains the frontend code, including JavaScript, HTML, and CSS, for rendering the web interface. The backend is handled by the `app.py` file, which is written in Flask.

## Future Work

- We could try a different way to connect neighboring neurons, either in a rectangular pattern or a circular one.
- We are thinking of making sub groups of puzzles based on the types and use different agent for each sub group. 

## How to contribute

Fork the repo make your changes and submit a pull request. Join our [Discord](https://discord.com/invite/nHuJc4Y4n7) and say hi:)

----------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------




# Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI)

This repository contains the ARC-AGI task data, as well as a browser-based interface for humans to try their hand at solving the tasks manually.

*"ARC can be seen as a general artificial intelligence benchmark, as a program synthesis benchmark, or as a psychometric intelligence test. It is targeted at both humans and artificially intelligent systems that aim at emulating a human-like form of general fluid intelligence."*

A complete description of the dataset, its goals, and its underlying logic, can be found in: [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547).

As a reminder, a test-taker is said to solve a task when, upon seeing the task for the first time, they are able to produce the correct output grid for *all* test inputs in the task (this includes picking the dimensions of the output grid). For each test input, the test-taker is allowed 3 trials (this holds for all test-takers, either humans or AI).


## Task file format

The `data` directory contains two subdirectories:

- `data/training`: contains the task files for training (400 tasks). Use these to prototype your algorithm or to train your algorithm to acquire ARC-relevant cognitive priors.
- `data/evaluation`: contains the task files for evaluation (400 tasks). Use these to evaluate your final algorithm. To ensure fair evaluation results, do not leak information from the evaluation set into your algorithm (e.g. by looking at the evaluation tasks yourself during development, or by repeatedly modifying an algorithm while using its evaluation score as feedback).

The tasks are stored in JSON format. Each task JSON file contains a dictionary with two fields:

- `"train"`: demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
- `"test"`: test input/output pairs. It is a list of "pairs" (typically 1 pair).

A "pair" is a dictionary with two fields:

- `"input"`: the input "grid" for the pair.
- `"output"`: the output "grid" for the pair.

A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 1x1 and the largest is 30x30.

When looking at a task, a test-taker has access to inputs & outputs of the demonstration pairs, plus the input(s) of the test pair(s). The goal is to construct the output grid(s) corresponding to the test input grid(s), using 3 trials for each test input. "Constructing the output grid" involves picking the height and width of the output grid, then filling each cell in the grid with a symbol (integer between 0 and 9, which are visualized as colors). Only *exact* solutions (all cells match the expected answer) can be said to be correct.


## Usage of the testing interface

The testing interface is located at `apps/testing_interface.html`. Open it in a web browser (Chrome recommended). It will prompt you to select a task JSON file.

After loading a task, you will enter the test space, which looks like this:

![test space](https://arc-benchmark.s3.amazonaws.com/figs/arc_test_space.png)

On the left, you will see the input/output pairs demonstrating the nature of the task. In the middle, you will see the current test input grid. On the right, you will see the controls you can use to construct the corresponding output grid.

You have access to the following tools:

### Grid controls

- Resize: input a grid size (e.g. "10x20" or "4x4") and click "Resize". This preserves existing grid content (in the top left corner).
- Copy from input: copy the input grid to the output grid. This is useful for tasks where the output consists of some modification of the input.
- Reset grid: fill the grid with 0s.

### Symbol controls

- Edit: select a color (symbol) from the color picking bar, then click on a cell to set its color.
- Select: click and drag on either the output grid or the input grid to select cells.
    - After selecting cells on the output grid, you can select a color from the color picking to set the color of the selected cells. This is useful to draw solid rectangles or lines.
    - After selecting cells on either the input grid or the output grid, you can press C to copy their content. After copying, you can select a cell on the output grid and press "V" to paste the copied content. You should select the cell in the top left corner of the zone you want to paste into.
- Floodfill: click on a cell from the output grid to color all connected cells to the selected color. "Connected cells" are contiguous cells with the same color.

### Answer validation

When your output grid is ready, click the green "Submit!" button to check your answer. We do not enforce the 3-trials rule.

After you've obtained the correct answer for the current test input grid, you can switch to the next test input grid for the task using the "Next test input" button (if there is any available; most tasks only have one test input).

When you're done with a task, use the "load task" button to open a new task.

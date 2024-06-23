import json
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from loguru import logger
from utils import generate_with_references, DEBUG
import datasets
from rich.console import Console
from rich.markdown import Markdown
from time import sleep
from utils import generate_with_references
from openai import OpenAI
clientDSC = OpenAI(api_key="sk-28xxxxxxx", base_url="https://api.deepseek.com")

default_reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]

console = Console()

def plot_grids(input_grid, output_grid, model_prediction_grid, title):
    """ Helper function to plot input, output, and transformed grids side by side """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(title)
    
    # Plot input grid
    im1 = axes[0].imshow(input_grid, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Input Grid')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot output grid
    im2 = axes[1].imshow(output_grid, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Output Grid')
    fig.colorbar(im2, ax=axes[1])
    
    # Plot transformed grid
    im3 = axes[2].imshow(model_prediction_grid, cmap='viridis', interpolation='nearest')
    axes[2].set_title('Transformed Grid')
    fig.colorbar(im3, ax=axes[2])
    
    plt.show()

def process_fn(item, full_dataset, temperature=0.7, max_tokens=2048):
    """ Processes a single item to find and apply the pattern """
    input_grid = item['input_grid']
    output_grid = item['output_grid']
    model = item['model']
    
    model_prediction_grid = find_and_apply_pattern(input_grid, output_grid, full_dataset)
    
    if DEBUG:
        logger.info(f"model: {model}, input_grid: {input_grid}, output_grid: {output_grid}, model_prediction_grid: {model_prediction_grid}")
    
    return {"model_prediction_grid": model_prediction_grid}

def apply_pattern_from_description(input_grid, pattern_description, model="Qwen/Qwen2-72B-Instruct", temperature=0.7, max_tokens=2048):
    """
    Applies the pattern described in the pattern_description to the input grid using the LLM.

    Args:
        input_grid (np.array): The input grid.
        pattern_description (str): The description of the pattern.
        model (str): The identifier of the model to use for generating the pattern application code.
        temperature (float): Controls the randomness of the response generation.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        np.array: The transformed grid after applying the pattern.
    """
    # Convert the input grid to a numpy array
    input_grid = np.array(input_grid)

    response = clientDSC.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates Python functions to apply patterns in grids."},
            {"role": "user", "content": f"Here is a numpy array named 'input_grid' that looks like this {input_grid}. Generate a Python function named 'apply_pattern' that applies the following pattern to 'input_grid': {pattern_description}. Write only the code without any additional text or formatting."}
        ],
        stream=False,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    function_code = response.choices[0].message.content.replace("```python", "").replace("```", "")
    print("function_code")
    print(function_code)
    print("-------------------------------------------------")
    print("==========================================================================")

    # Strip any leading/trailing whitespace and newlines
    function_code = function_code.strip()

    # Define a local function using the generated code
    local_vars = {}
    exec(function_code, globals(), local_vars)
    apply_pattern = local_vars.get('apply_pattern')

    if not callable(apply_pattern):
        raise ValueError("Generated code does not define a callable 'apply_pattern' function.")

    # Apply the pattern using the generated function
    model_prediction_grid = apply_pattern(input_grid)

    return model_prediction_grid

def find_and_apply_pattern(input_grid, output_grid, full_dataset, model="Qwen/Qwen2-72B-Instruct", temperature=0.7, max_tokens=2048):
    """
    Finds the pattern between the input and output grids using an LLM and applies it to the input grid.

    Args:
        input_grid (np.array): The input grid.
        output_grid (np.array): The output grid.
        full_dataset (str): the full grids.json.
        model (str): The identifier of the model to use for generating the pattern description.
        temperature (float): Controls the randomness of the response generation.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        np.array: The transformed grid after applying the pattern.
    """
    # Ensure the input and output grids are numpy arrays
    input_grid = np.array(input_grid)
    output_grid = np.array(output_grid)

    # Check if the grids have the same shape
    if input_grid.shape != output_grid.shape:
        raise ValueError("Input and output grids must have the same shape.")

    pattern_descriptions = []
    for llm_model in default_reference_models:
        # Generate a description of the pattern using the LLM
        pattern_descriptions.append(generate_with_references(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that describes patterns in grids."},
                    {"role": "user", "content": f"What do you see in the inputs? What is the pattern that leads to the output? Describe the pattern between the input grids and their relative output: {full_dataset}"}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    print("==========================================================================")
    print("-------------------------------------------------")
    print("pattern_description")
    print(pattern_descriptions)
    print("-------------------------------------------------")
    final_pattern_description = generate_with_references(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that describes patterns in grids."},
                {"role": "user", "content": f"Describe the pattern between the input grids and their relative output: {full_dataset}, here what the other assistants found searching: {pattern_descriptions}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    print("final_pattern_description")
    print(final_pattern_description)
    print("-------------------------------------------------")

    # Apply the pattern based on the description
    model_prediction_grid = apply_pattern_from_description(input_grid, final_pattern_description, model=model, temperature=temperature, max_tokens=max_tokens)

    return model_prediction_grid

def main():
    # Load the JSON data
    with open('grids.json') as f:
        data = json.load(f)
    
    # Iterate through the 'train' dataset to find the pattern
    for idx, example in enumerate(data['train']):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        full_dataset = data['train']

        # Prepare data for the MoA system
        moa_data = {
            "input_grid": [input_grid.tolist()],
            "output_grid": [output_grid.tolist()],
            "model": ["Qwen/Qwen2-72B-Instruct"]  # Replace with the actual model identifier
        }
        
        # Process the data with the MoA system
        eval_set = datasets.Dataset.from_dict(moa_data)
        eval_set = eval_set.map(
            partial(process_fn, temperature=0.7, max_tokens=2048, full_dataset=full_dataset),
            batched=False,
            num_proc=1,
        )
        
        # Extract the transformed grid
        model_prediction_grid = np.array(eval_set[0]['model_prediction_grid'])
        
        # Plot the grids
        plot_grids(input_grid, output_grid, model_prediction_grid, f'Train Example {idx + 1}')

    # Apply the pattern to the test grid
    test_grid = np.array(data['test'][0]['input'])
    final_pattern_description = generate_with_references(
        model=default_reference_models[0],
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes patterns in grids."},
            {"role": "user", "content": f"Describe the pattern between the input grids and their relative output: {data['train']}"}
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    test_prediction_grid = apply_pattern_from_description(test_grid, final_pattern_description)
    plot_grids(test_grid, np.array(data['test'][0]['output']), test_prediction_grid, 'Test Example')

if __name__ == "__main__":
    main()
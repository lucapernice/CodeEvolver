To run the code evolution process, you'll need to configure the config.yaml file and then execute the main.py script.
Make sure python>=3.10 and install requirements from requirements.txt

### 1. Modify config.yaml

The config.yaml file controls various parameters for the evolution process. Open config.yaml and adjust the following settings as needed:

*   **`c_file_path`**: Path to your initial C source code file (e.g., `"compression.c"`).
*   **`model`**: The LLM model you want to use. Examples are provided in the comments.
*   **`is_reasoning`**: Set to `True` if your chosen model supports reasoning and you want to enable it.
*   **`temperature`**: Controls the randomness of the LLM's output. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more deterministic.
*   **`max_tokens`**: The maximum number of tokens the LLM can generate in a single response.
*   **`population_size`**: The number of individuals (code variants) in each generation. A larger population can explore more possibilities but will take longer to process.
*   **`generations`**: The number of evolutionary cycles to run. More generations can lead to better solutions but increase runtime.
*   **logs**: Set to `True` to enable detailed logging (can produce large log files). CSV logging of fitness scores will occur regardless of this setting.
*   **`test_files`**: A list of file paths used to evaluate the performance of the generated C code. Ensure these files exist and are accessible.
*   **`evolved_code_output_path`**: The file path where the best-evolved C code will be saved.

**Example config.yaml snippet:**

````yaml
# ...existing code...
c_file_path: "my_custom_compression.c"
model: "meta-llama/Llama-3.3-70B-Instruct"
population_size: 10
generations: 5
test_files:
  - "dataset.txt"
  - "dataset2.txt" # Ensure this path is correct
evolved_code_output_path: "output/best_evolved_code.c"
# ...existing code...
````

**Important:**
*   Ensure your Nebius API key is set as an environment variable `NEBIUS_API_KEY`. You can typically do this by creating a .env file in the NLP_project directory with the content:
    ```
    NEBIUS_API_KEY="your_api_key_here"
    ```
    The main.py script will load this using `load_dotenv()`.

### 2. Run the Evolution

Once you have configured config.yaml and set up your environment variables, you can run the evolution process by executing the main.py script from your terminal within the NLP_project directory:

```bash
python main.py
```

The script will:
1.  Load the configuration from config.yaml.
2.  Initialize the `CodeEvolver`.
3.  Run the evolutionary algorithm, which involves generating, compiling, and testing C code variants.
4.  Save the best-evolved code to the path specified in `evolved_code_output_path`.
5.  Save the evolution path in a csv file in 'logs/'

You will see output in the terminal indicating the progress of the evolution, including fitness scores for each individual and generation.
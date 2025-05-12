import yaml
import asyncio
from dotenv import load_dotenv
from utils.code_evolver import CodeEvolver
from utils.logs_manager import LogManager

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return None
    
async def main():
    # Load environment variables (e.g., for NEBIUS_API_KEY if not hardcoded)
    load_dotenv()

    # Load configuration
    config = load_config()
    if not config:
        print("Exiting due to configuration error.")
        return

    # Extract parameters from config, providing defaults if necessary
    c_file_path = config.get("c_file_path", "compression.c")
    model = config.get("model", "meta-llama/Llama-3.3-70B-Instruct")
    is_reasoning = config.get("is_reasoning", False)
    temperature = config.get("temperature", 0.6)
    max_tokens = config.get("max_tokens", 10000)
    population_size = config.get("population_size", 20)
    generations = config.get("generations", 10)
    logs = config.get("logs", False) # Default to False as per your config comment
    test_files = config.get("test_files", ["dataset2.txt"])
    evolved_code_output_path = config.get("evolved_code_output_path", "evolved_compression.c")
    tournament_k = config.get("tournament_k", 3) # Read tournament_k

    # Initialize CodeEvolver
    print(f"Initializing CodeEvolver with C file: {c_file_path}")
    code_evolver = CodeEvolver(
        c_file_path=c_file_path,
        model=model,
        is_reasoning=is_reasoning,
        temperature=temperature,
        max_tokens=max_tokens,
        population_size=population_size,
        generations=generations,
        logs=logs,
        tournament_k=tournament_k # Pass tournament_k
    )

    # Run the evolution
    print("Starting asynchronous evolution...")
    try:
        evolved_code = await code_evolver.run_evolution()
    except Exception as e:
        print(f"An error occurred during evolution: {e}")
        evolved_code = None

    # Write the evolved code to a file
    if evolved_code:
        try:
            code_evolver.write_file(evolved_code_output_path, evolved_code)
            print(f"Evolved code written to '{evolved_code_output_path}'")
        except Exception as e:
            print(f"Error writing evolved code to file: {e}")
    else:
        print("No evolved code generated or an error occurred.")

if __name__ == "__main__":
    asyncio.run(main())
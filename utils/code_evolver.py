import re
import json
import os
import random
import subprocess
import time
import yaml
from typing import Any, Dict
from openai import AsyncOpenAI 
from dotenv import load_dotenv
from utils.logs_manager import LogManager
import asyncio 


class CodeEvolver:
    def __init__(self, c_file_path:str, model:str, is_reasoning:bool = False, temperature:float = 0.6, max_tokens:int = 10000, population_size:int =10, generations:int =2, logs:bool = True, tournament_k:int = 3, test_files:list[str] | None = None): # Added tournament_k
        self.c_file_path = c_file_path
        self.population_size = population_size
        self.generations = generations
        self.log_manager = LogManager(active=logs)
        self.original_code = self.read_file(c_file_path)
        self.best_fitness = 0
        self.best_individual = None  
        self.model = model
        self.reasoning = is_reasoning
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tournament_k = tournament_k # Store tournament size
        self.max_retries = 3
        self.retry_base_delay = 1.5
        self.response_schema_version = "1.0"
        self.response_format: Any = {
            "type": "json_schema",
            "json_schema": {
                "name": "compression_evolution_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["schema_version", "full_c_code", "functions"],
                    "properties": {
                        "schema_version": {"type": "string"},
                        "full_c_code": {"type": "string", "minLength": 1},
                        "functions": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["compress", "decompress"],
                            "properties": {
                                "compress": {"type": "string", "minLength": 1},
                                "decompress": {"type": "string", "minLength": 1}
                            }
                        },
                        "notes": {"type": "string"},
                        "strategy": {"type": "string"}
                    }
                }
            }
        }

        # Modifica: Inizializza il client AsyncOpenAI una volta
        load_dotenv()
        self.async_client = AsyncOpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.getenv("NEBIUS_API_KEY")
        )
        # Mantieni anche il client sincrono se serve altrove (anche se non sembra in questo codice)
        # self.sync_client = OpenAI(...)

        print(f'Using the model: {self.model}')
        print(f"Reasoning mode: {'Enabled' if self.reasoning else 'Disabled'}")
        
        
        # Test files to evaluate the compression algorithm
        self.test_files = test_files if test_files else ["data/dataset2.txt"]

    def _default_metrics(self):
        return {
            'compression_ratio': 1.0,
            'compression_time': 999,
            'decompression_time': 999,
            'integrity_check': False,
            'fitness': 0
        }

    async def _call_llm_with_retry(self, messages, operation_label):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.log_manager.add_log(f'LLM call ({operation_label}) attempt {attempt}/{self.max_retries}')
                return await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=self.response_format
                )
            except Exception as error:
                last_error = error
                wait_seconds = self.retry_base_delay * (2 ** (attempt - 1))
                self.log_manager.add_log(f'LLM call failed ({operation_label}) attempt {attempt}: {error}')
                if attempt < self.max_retries:
                    self.log_manager.add_log(f'Waiting {wait_seconds:.1f}s before retry ({operation_label})')
                    await asyncio.sleep(wait_seconds)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"LLM call failed without explicit exception ({operation_label})")

    def read_file(self, file_path):
        """Reads the content of a file"""
        with open(file_path, 'r') as file:
            self.log_manager.add_log(f'Read file {file_path}')
            return file.read()
        
    def write_file(self, file_path, content):
        """Writes content to a file"""
        with open(file_path, 'w') as file:
            self.log_manager.add_log(f'Write to file {file_path}')
            file.write(content)
    
    def create_llm_prompt(self, current_code: str):
        """Creates the prompt for the LLM"""
        self.log_manager.add_log(f'Creating LLM prompt for generation')
        yaml_file = 'utils/prompts.yaml'
        with open(yaml_file, 'r') as file:
            prompts = yaml.safe_load(file)
        prompt = prompts['premise']
        current = prompts['current_code'].format(full_code=current_code)
        prompt += current
        tasks_variations_list = ['base', 'complex', 'uncommon']
        #select random task
        task_variation = random.choice(tasks_variations_list)
        creation_task = prompts['creation_task'][task_variation]
        prompt += creation_task
        formatting = prompts['format_json'].format(schema_version=self.response_schema_version)
        prompt += formatting
        return prompt

    def _validate_llm_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.log_manager.add_log('Validating JSON payload from LLM response')
        if not isinstance(payload, dict):
            raise ValueError('LLM payload must be a JSON object')

        required_keys = {'schema_version', 'full_c_code', 'functions'}
        missing_keys = required_keys.difference(payload.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in LLM JSON payload: {sorted(missing_keys)}")

        schema_version = payload.get('schema_version')
        if schema_version != self.response_schema_version:
            raise ValueError(
                f"Unsupported schema_version '{schema_version}', expected '{self.response_schema_version}'"
            )

        full_c_code = payload.get('full_c_code')
        if not isinstance(full_c_code, str) or not full_c_code.strip():
            raise ValueError('full_c_code must be a non-empty string')

        functions = payload.get('functions')
        if not isinstance(functions, dict):
            raise ValueError('functions must be a JSON object')

        compress_fn = functions.get('compress')
        decompress_fn = functions.get('decompress')
        if not isinstance(compress_fn, str) or not compress_fn.strip():
            raise ValueError('functions.compress must be a non-empty string')
        if not isinstance(decompress_fn, str) or not decompress_fn.strip():
            raise ValueError('functions.decompress must be a non-empty string')

        if 'compress(' not in compress_fn or 'decompress(' not in decompress_fn:
            raise ValueError('functions object does not contain valid compress/decompress signatures')

        if 'compress(' not in full_c_code or 'decompress(' not in full_c_code:
            raise ValueError('full_c_code does not appear to contain compress/decompress functions')

        return payload

    def parse_llm_json_payload(self, llm_response: str) -> Dict[str, Any]:
        self.log_manager.add_log('Parsing LLM response as strict JSON payload')
        content = llm_response or ''
        if self.reasoning and '</think>' in content:
            content = content.split('</think>', 1)[1]

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(f'Invalid JSON response from LLM: {error}') from error

        return self._validate_llm_payload(payload)

    def _validate_candidate_c_code(self, code: str) -> None:
        """Performs lightweight sanity checks before compilation.

        Raises ValueError if the candidate looks malformed.
        """
        if not isinstance(code, str) or not code.strip():
            raise ValueError('Candidate C code is empty')

        required_snippets = ['#include', 'int main(', 'compress(', 'decompress(']
        missing = [snippet for snippet in required_snippets if snippet not in code]
        if missing:
            raise ValueError(f'Candidate C code missing required elements: {missing}')

        suspicious_patterns = [
            r"printf\s*\(\s*'",
            r"fprintf\s*\(\s*'",
            r"snprintf\s*\([^\)]*'",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, code):
                raise ValueError(f'Candidate C code contains suspicious pattern: {pattern}')

        if code.count('\n') < 20:
            raise ValueError('Candidate C code appears abnormally compressed on a single line')
    

    # Modifica: Rendi la funzione async
    async def evolve_functions_with_llm(self, current_code: str, generation: int, individual: int, feedback: Any = None) -> str:
        """Evolves full C code using an LLM (asynchronously)."""
        self.log_manager.add_log(f'Evolving functions with LLM for generation {generation}, individual {individual}')
        prompt = self.create_llm_prompt(current_code)
        
        try:
            self.log_manager.add_log(f'Calling LLM asynchronously...')
            response = await self._call_llm_with_retry(
                [
                    {"role": "system", "content": "You are an expert C programmer specializing in compression algorithms."},
                    {"role": "user", "content": prompt}
                ],
                operation_label=f'generation_{generation}_individual_{individual}'
            )
            self.log_manager.add_log(f'LLM response: \n\n\n {response.choices[0].message.content} \n\n\n')
            
            evolved_code = response.choices[0].message.content or ""
            payload = self.parse_llm_json_payload(evolved_code)
            candidate_code = payload['full_c_code']
            self._validate_candidate_c_code(candidate_code)
            return candidate_code
        except Exception as e:
            print(f"Error in LLM call: {e}")
            self.log_manager.add_log(f'Error in LLM call: {e}')
            return current_code
        
    def parse_test_output(self, output):
        """Parse test program output to extract metrics"""
        self.log_manager.add_log(f'Parsing test output: \n\n\n {output} \n\n\n')
        metrics = {}
        
        try:
            # Extract compression ratio
            self.log_manager.add_log(f'Parsing test output: Rapporto di compressione')
            ratio_match = re.search(r'Rapporto di compressione: ([0-9.]+)', output)
            if ratio_match:
                metrics['compression_ratio'] = float(ratio_match.group(1)) 
                self.log_manager.add_log(f'Compression ratio: {ratio_match.group(1)}')               
            
            # Extract times
            self.log_manager.add_log(f'Parsing test output: Tempo di compressione e decompressione')
            c_time_match = re.search(r'Tempo di compressione: ([0-9.]+)', output)
            if c_time_match:
                metrics['compression_time'] = float(c_time_match.group(1))
                self.log_manager.add_log(f'Compression time: {c_time_match.group(1)}')
            
            self.log_manager.add_log(f'Parsing test output: Tempo di decompressione')
            d_time_match = re.search(r'Tempo di decompressione: ([0-9.]+)', output)
            if d_time_match:
                metrics['decompression_time'] = float(d_time_match.group(1))
                self.log_manager.add_log(f'Decompression time: {d_time_match.group(1)}')
            
            # Extract integrity check
            self.log_manager.add_log(f'Parsing test output: Controllo integrità')
            integrity_match = re.search(r'Controllo integrità: (SUCCESSO|FALLITO)', output)
            if integrity_match:
                metrics['integrity_check'] = integrity_match.group(1) == "SUCCESSO"
                self.log_manager.add_log(f'Integrity check: {integrity_match.group(1)}')
            
            # Extract fitness score
            self.log_manager.add_log(f'Parsing test output: Punteggio fitness')
            fitness_match = re.search(r'Punteggio fitness: ([0-9.]+)', output)
            if fitness_match:
                metrics['fitness'] = float(fitness_match.group(1))
                self.log_manager.add_log(f'Fitness score: {fitness_match.group(1)}')
        
            return {
                'compression_ratio': metrics.get('compression_ratio', 1.0),
                'compression_time': metrics.get('compression_time', 999),
                'decompression_time': metrics.get('decompression_time', 999),
                'integrity_check': metrics.get('integrity_check', False),
                'fitness': metrics.get('fitness', 0)
            }
        except Exception as e:
            print(f"Error parsing output: {e}")
            self.log_manager.add_log(f'Error parsing output {e}\n Returning default metrics')
            return self._default_metrics()
        
    def compile_and_test(self, code):
        """Compiles C code and tests it with test files"""
        self.log_manager.add_log(f'Compiling and testing code.')
        # Write code to a temporary file
        temp_file = "temp_compression.c"
        compiled_file = "temp_compression"
        self.write_file(temp_file, code)
        
        # Compile the code
        self.log_manager.add_log(f'Compiling code')
        compile_cmd = f"gcc {temp_file} -o {compiled_file}"
        try:
            subprocess.run(compile_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            self.log_manager.add_log(f'Compilation error')
            print("Compilation error")
            return None
        
        # Run tests on different files
        self.log_manager.add_log(f'Running tests')
        results = []
        for test_file in self.test_files:
            self.log_manager.add_log(f'Running test on file {test_file}')
            cmd = f"./{compiled_file} {test_file}"
            try:
                # Set a timeout for the subprocess (e.g., 10 seconds)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                self.log_manager.add_log(f'Test output: \n\n\n {result.stdout} \n\n\n') 
                print(result.stdout)
                results.append(self.parse_test_output(result.stdout))                
            except subprocess.TimeoutExpired:
                self.log_manager.add_log(f'Error: Test on {test_file} timed out after 10 seconds')
                print(f"Error: Test on {test_file} timed out after 10 seconds")
                # Add a failed result with zero fitness
                results.append(self._default_metrics())
            except Exception as e:
                self.log_manager.add_log(f'Error running test on {test_file}: {e}')
                print(f"Error running test on {test_file}: {e}")
                # Add a failed result with zero fitness
                results.append(self._default_metrics())
        
        # Clean up temporary files
        try:
            os.remove(temp_file)
            os.remove(compiled_file)
        except:
            pass
        
        # Calculate average fitness
        if results:
            try:
                self.log_manager.add_log(f'Calculating average fitness')
                avg_fitness = sum(r.get('fitness', 0) for r in results) / len(results)
                avg_ratio = sum(r.get('compression_ratio', 1.0) for r in results) / len(results)
                avg_c_time = sum(r.get('compression_time', 999) for r in results) / len(results)
                avg_d_time = sum(r.get('decompression_time', 999) for r in results) / len(results)
                integrity = all(r.get('integrity_check', False) for r in results)
                self.log_manager.add_log(f'Average fitness: {avg_fitness}, Compression ratio: {avg_ratio}, Compression time: {avg_c_time}, Decompression time: {avg_d_time}, Integrity check: {integrity}')

                return {
                    'fitness': avg_fitness,
                    'compression_ratio': avg_ratio,
                    'compression_time': avg_c_time,
                    'decompression_time': avg_d_time,
                    'integrity_check': integrity
                }
            except Exception as e:
                self.log_manager.add_log(f'Error calculating average fitness: {e}')
                print(f"Error calculating average fitness: {e}")
                return self._default_metrics()
        else:
            self.log_manager.add_log(f'No results from tests')
            return None

    
    def _tournament_selection(self, fitness_scores_with_indices):
        """
        Selects an individual's data using tournament selection.
        Args:
            fitness_scores_with_indices: List of dicts {'individual_index': i, 'metrics': results}.
                                         Assumes 'fitness' is a key in 'metrics'.
        Returns:
            A dictionary {'individual_index': index, 'metrics': metrics} for the winning parent.
            Returns None if selection cannot be performed (e.g., empty input or problematic data).
        """
        if not fitness_scores_with_indices:
            self.log_manager.add_log("Tournament selection error: Input 'fitness_scores_with_indices' is empty.")
            return None

        num_candidates = len(fitness_scores_with_indices)
        actual_tournament_size = min(self.tournament_k, num_candidates)

        if actual_tournament_size <= 0:
            self.log_manager.add_log(f"Tournament selection error: Effective tournament size is {actual_tournament_size}, num_candidates: {num_candidates}.")
            return None
        
        try:
            participants_data = random.sample(fitness_scores_with_indices, actual_tournament_size)
        except ValueError as e: 
            self.log_manager.add_log(f"Tournament selection error during random.sample: {e}. Candidates: {num_candidates}, Size: {actual_tournament_size}")
            return None

        try:
            for p in participants_data:
                if 'metrics' not in p or 'fitness' not in p['metrics']:
                    self.log_manager.add_log(f"Tournament selection error: Participant missing 'metrics' or 'fitness'. Data: {p}")
                    return None 

            winner_data = max(participants_data, key=lambda x: x['metrics']['fitness'])
        except (ValueError, KeyError, TypeError) as e: 
            self.log_manager.add_log(f"Tournament selection error during winner determination: {e}. Participants: {participants_data}")
            return None
            
        return winner_data

    # Modifica: Rendi la funzione async
    async def mutate_crossover(self, parent1, parent1_metrics, parent2=None, parent2_metrics=None):
        """Apply mutation or crossover to two individuals using LLM (asynchronously)"""
        operation = "mutation" if parent2 is None else "crossover"
        self.log_manager.add_log(f'Performing {operation} between individuals')       
        yaml_file = 'utils/prompts.yaml'
        with open(yaml_file, 'r') as file:
            prompts = yaml.safe_load(file)        
        
        if parent2 is not None:
            prompt = prompts['premise']
            if parent2_metrics is None:
                parent2_metrics = parent1_metrics
            # Add parent metrics to the prompt
            parent1_feedback = f"Parent 1 Metrics: Fitness: {parent1_metrics['fitness']:.2f}, Compression Ratio: {parent1_metrics['compression_ratio']:.2f}, Integrity: {parent1_metrics['integrity_check']}"
            parent2_feedback = f"Parent 2 Metrics: Fitness: {parent2_metrics['fitness']:.2f}, Compression Ratio: {parent2_metrics['compression_ratio']:.2f}, Integrity: {parent2_metrics['integrity_check']}"
            prompt += f"\n{parent1_feedback}\n{parent2_feedback}\n"

            tasks_variations_list = ['base', 'complex']
            #select random task
            task_variation = random.choice(tasks_variations_list)
            prompt += prompts['crossover_task'][task_variation].format(
                parent1_code=parent1,
                parent2_code=parent2
            ) 
            formatting = prompts['format_json'].format(schema_version=self.response_schema_version)
            prompt += formatting

        else:  # mutation
            prompt = self.create_llm_prompt(parent1)
            # Add parent metrics to the prompt
            parent_feedback = f"Parent Metrics: Fitness: {parent1_metrics['fitness']:.2f}, Compression Ratio: {parent1_metrics['compression_ratio']:.2f}, Integrity: {parent1_metrics['integrity_check']}"
            prompt = f"{parent_feedback}\n{prompt}"
                    
        try:
            
            self.log_manager.add_log(f'Calling LLM asynchronously for {operation}...')
            response = await self._call_llm_with_retry(
                [
                    {"role": "system", "content": "You are an expert in genetic algorithms and data compression. Return strict JSON only."},
                    {"role": "user", "content": prompt}
                ],
                operation_label=operation
            )
            
            evolved_code = response.choices[0].message.content or ""
            self.log_manager.add_log(f'LLM response: \n\n\n {evolved_code} \n\n\n')
            payload = self.parse_llm_json_payload(evolved_code)
            new_individual = payload['full_c_code']
            self._validate_candidate_c_code(new_individual)
            self.log_manager.add_log('New individual generated from full_c_code payload')
            return new_individual
        except Exception as e:
            print(f"Error in genetic operation: {e}")
            self.log_manager.add_log(f'Error in genetic operation: {e}, returning original individual')
            return parent1  # Return first parent if there's an error

    # Modifica: Rendi la funzione async
    async def run_evolution(self):
        """Runs the complete evolutionary algorithm (asynchronously)"""
        print("Starting compression algorithm evolution...")
        self.log_manager.add_log(f'Starting compression algorithm evolution...')
        self.log_manager.add_log(f'Model: {self.model}, Temperature: {self.temperature}, Reasoning: {self.reasoning}, Tournament K: {self.tournament_k}')
        
        # Create initial population
        population = [self.original_code]
        
        # Generate initial variants using LLM (Asincrono)
        print(f"Generating initial {self.population_size - 1} individuals asynchronously...")
        self.log_manager.add_log(f'Generating initial {self.population_size - 1} individuals asynchronously...')
        tasks = []
        for i in range(1, self.population_size):
            tasks.append(self.evolve_functions_with_llm(self.original_code, 0, i))

        evolved_functions_list = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(evolved_functions_list, 1): # enumerate starts from 1 to match individual 'i'
            if isinstance(result, Exception):
                print(f"Error generating initial individual {i}: {result}")
                self.log_manager.add_log(f'Error generating initial individual {i}: {result}, adding copy of original code')
                population.append(self.original_code)
            elif isinstance(result, str):
                if result.strip():
                    population.append(result)
                else:
                    print(f"Empty code for initial individual {i}, adding copy of original code")
                    self.log_manager.add_log(f'Empty code for initial individual {i}, adding copy of original code')
                    population.append(self.original_code)
            else:
                print(f"Error extracting functions for initial individual {i}, adding copy of original code")
                self.log_manager.add_log(f'Error extracting functions for initial individual {i}, adding copy of original code')
                population.append(self.original_code)
        
        while len(population) < self.population_size:
            self.log_manager.add_log(f'Initial population smaller than expected ({len(population)}/{self.population_size}), adding copy of original code.')
            population.append(self.original_code)
        population = population[:self.population_size]


        # Evaluate fitness of initial population (Sincrono)
        fitness_scores = []
        for i, individual_code in enumerate(population): 
            try:
                print(f"Evaluating initial individual {i}/{len(population)-1}...")
                self.log_manager.add_log(f'Evaluating initial individual {i}/{len(population)-1}...')
                if not isinstance(individual_code, str):
                    self.log_manager.add_log(f"Error: Individual {i} is not a string. Type: {type(individual_code)}. Using original code.")
                    individual_code = self.original_code
                    population[i] = self.original_code

                fitness_results = self.compile_and_test(individual_code)

                if fitness_results:
                    fitness_scores.append({'individual_index': i, 'metrics': fitness_results})
                    print(f"  Fitness: {fitness_results['fitness']:.2f}, Ratio: {fitness_results['compression_ratio']:.2f}, Integrity: {fitness_results['integrity_check']}")
                    
                    if fitness_results['fitness'] > self.best_fitness and fitness_results['integrity_check']:
                        self.log_manager.add_log(f'New best individual found (initial population)')
                        self.best_fitness = fitness_results['fitness']
                        self.best_individual = individual_code
                else: 
                    self.log_manager.add_log(f"Initial individual {i} failed compilation/testing. Assigning default low fitness.")
                    default_metrics = self._default_metrics()
                    fitness_scores.append({'individual_index': i, 'metrics': default_metrics})
                    fitness_results = default_metrics

            except Exception as e:
                print(f"Error evaluating initial individual {i}: {e}")
                self.log_manager.add_log(f'Error evaluating initial individual {i}: {e}')
                population[i] = self.original_code 
                default_metrics = self._default_metrics()
                fitness_scores.append({'individual_index': i, 'metrics': default_metrics})
                fitness_results = default_metrics 
            
            if not fitness_results: 
                fitness_results = self._default_metrics()

            csv_row = { 
                'generation': 0, 
                'individual': i, 
                'fitness': fitness_results.get('fitness', 0), 
                'compression_ratio': fitness_results.get('compression_ratio', 1.0), 
                'integrity_check': fitness_results.get('integrity_check', False) 
            }
            self.log_manager.add_csv_row(csv_row)
        
        
        # Evolution for specified generations
        for gen in range(1, self.generations + 1):
            print(f"\n=== Generation {gen}/{self.generations} ===")
            self.log_manager.add_log(f'=== Generation {gen}/{self.generations} ===')
            
            if not fitness_scores:
                self.log_manager.add_log(f"Generation {gen}: No individuals with fitness scores from previous generation. Stopping evolution.")
                print(f"Generation {gen}: No individuals with fitness scores. Stopping evolution.")
                break 

            fitness_scores.sort(key=lambda x: x['metrics']['fitness'], reverse=True)
            
            evolution_tasks = []
            num_individuals_to_generate = self.population_size
            
            generated_tasks_count = 0
            original_code_fallback_metrics = self._default_metrics()

            while generated_tasks_count < num_individuals_to_generate:
                parent1_data = self._tournament_selection(fitness_scores)
                
                if parent1_data is None:
                    self.log_manager.add_log(f"Generation {gen}: Parent 1 selection failed. Adding mutation of original code as a task.")
                    evolution_tasks.append(self.mutate_crossover(self.original_code, original_code_fallback_metrics))
                    generated_tasks_count += 1
                    continue

                parent1_code = population[parent1_data['individual_index']]
                parent1_metrics = parent1_data['metrics']

                if random.random() < 0.7 and self.population_size > 1: 
                    parent2_data = self._tournament_selection(fitness_scores)
                    if parent2_data is None or parent2_data['individual_index'] == parent1_data['individual_index']:
                        self.log_manager.add_log(f"Generation {gen}: Parent 2 selection failed or same as Parent 1. Falling back to mutation of Parent 1 (idx: {parent1_data['individual_index']}).")
                        evolution_tasks.append(self.mutate_crossover(parent1_code, parent1_metrics))
                    else:
                        parent2_code = population[parent2_data['individual_index']]
                        parent2_metrics = parent2_data['metrics']
                        self.log_manager.add_log(f"Creating new individual by crossover (Gen {gen}) between P1_idx:{parent1_data['individual_index']} and P2_idx:{parent2_data['individual_index']}")
                        evolution_tasks.append(self.mutate_crossover(parent1_code, parent1_metrics, parent2_code, parent2_metrics))
                else: 
                    self.log_manager.add_log(f"Creating new individual by mutation (Gen {gen}) from P_idx:{parent1_data['individual_index']}")
                    evolution_tasks.append(self.mutate_crossover(parent1_code, parent1_metrics))
                
                generated_tasks_count += 1
            
            new_population_individuals = []
            if evolution_tasks:
                evolved_children_results = await asyncio.gather(*evolution_tasks, return_exceptions=True)
                for i, child_result in enumerate(evolved_children_results):
                    if isinstance(child_result, Exception):
                        self.log_manager.add_log(f'Error creating child {i} in Gen {gen} (asyncio.gather): {child_result}, adding copy of original')
                        new_population_individuals.append(self.original_code) 
                    elif child_result and isinstance(child_result, str): 
                        new_population_individuals.append(child_result)
                    else: 
                        self.log_manager.add_log(f'Failed to create child {i} in Gen {gen} (result type: {type(child_result)}), adding copy of original')
                        new_population_individuals.append(self.original_code) 

            while len(new_population_individuals) < self.population_size:
                self.log_manager.add_log(f'Population too small after evolution (Gen {gen}, size {len(new_population_individuals)}), adding copy of original code')
                new_population_individuals.append(self.original_code)

            population = new_population_individuals[:self.population_size] 
            
            current_gen_fitness_scores = []
            for i, individual_code in enumerate(population):
                print(f"Evaluating Gen {gen} individual {i}/{len(population)-1}...")
                try:
                    self.log_manager.add_log(f'Evaluating Gen {gen} individual {i}/{len(population)-1}...')
                    if not isinstance(individual_code, str): 
                        self.log_manager.add_log(f"Error: Gen {gen} Individual {i} is not a string. Type: {type(individual_code)}. Using original code.")
                        individual_code = self.original_code
                        population[i] = self.original_code 

                    fitness_results = self.compile_and_test(individual_code)
                    if fitness_results:
                        current_gen_fitness_scores.append({'individual_index': i, 'metrics': fitness_results})
                        print(f"  Fitness: {fitness_results['fitness']:.2f}, Ratio: {fitness_results['compression_ratio']:.2f}, Integrity: {fitness_results['integrity_check']}")
                        
                        if fitness_results['fitness'] > self.best_fitness and fitness_results['integrity_check']:
                            self.log_manager.add_log(f'New best individual found in Gen {gen}, Individual {i}')
                            self.best_fitness = fitness_results['fitness']
                            self.best_individual = individual_code
                    else: 
                        self.log_manager.add_log(f"Gen {gen} individual {i} failed compilation/testing. Assigning default low fitness.")
                        default_metrics = self._default_metrics()
                        current_gen_fitness_scores.append({'individual_index': i, 'metrics': default_metrics})
                        fitness_results = default_metrics 
                except Exception as e:
                    print(f"Error evaluating Gen {gen} individual {i}: {e}")
                    self.log_manager.add_log(f'Error evaluating Gen {gen} individual {i}: {e}')
                    population[i] = self.original_code
                    default_metrics = self._default_metrics()
                    current_gen_fitness_scores.append({'individual_index': i, 'metrics': default_metrics})
                    fitness_results = default_metrics
                
                if not fitness_results: 
                    fitness_results = self._default_metrics()
                
                csv_row = { 
                    'generation': gen, 
                    'individual': i, 
                    'fitness': fitness_results.get('fitness',0), 
                    'compression_ratio': fitness_results.get('compression_ratio',1.0), 
                    'integrity_check': fitness_results.get('integrity_check',False) 
                }
                self.log_manager.add_csv_row(csv_row)
            
            fitness_scores = current_gen_fitness_scores

            print(f"\nBest fitness so far (End of Gen {gen}): {self.best_fitness:.2f}")
            self.log_manager.add_log(f'Best fitness so far (End of Gen {gen}): {self.best_fitness:.2f}')
        
        if self.best_individual:
            print("\nEvolution completed!")            
            print(f"Best fitness achieved: {self.best_fitness:.2f}")
            self.log_manager.add_log(f'Evolution completed! Best fitness achieved: {self.best_fitness:.2f}')
            return self.best_individual
        else:
            print("\nNo improvements found. Returning original code.")
            self.log_manager.add_log(f'No improvements found. Returning original code.')
            return self.original_code
    



if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    models = {'Deepsick R1':'deepseek-ai/DeepSeek-R1','DeepSeek V3-0324':'deepseek-ai/DeepSeek-V3-0324', 'LLama 3.3':'meta-llama/Llama-3.3-70B-Instruct'}

    # Initialize CodeEvolver
    code_evolver = CodeEvolver("src/c/compression.c", model = models['LLama 3.3'],is_reasoning=False,temperature = 0.6 ,population_size=10, generations=1, logs = False, tournament_k=3) # Added tournament_k to instantiation
    
    # Modifica: Usa asyncio.run per eseguire la funzione async
    print("Starting asynchronous evolution...")
    evolved_code = asyncio.run(code_evolver.run_evolution())

    # Write the evolved code to a file
    if evolved_code:
        code_evolver.write_file("outputs/evolved/evolved_compression.c", evolved_code)
        print("Evolved code written to 'outputs/evolved/evolved_compression.c'")
    else:
        print("No evolved code generated.")

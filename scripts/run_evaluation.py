import os
import json
import glob
import time
import statistics
import requests
import re
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# =======================================================================
# SCRIPT CONFIGURATION
# =======================================================================

# --- Model and API Configuration ---
# The API key is loaded securely from an environment variable.
# Before running, execute: export GOOGLE_API_KEY="your-api-key"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-1.5-pro-latest"  # Using the latest stable model
GEMINI_ENDPOINT_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# --- Default File Paths (relative to the repository root) ---
DEFAULT_INFERENCE_DIR = "results/inference_outputs"
DEFAULT_DATA_DIR = "data"
DEFAULT_EVAL_OUTPUT_DIR = "results/evaluation_scores"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"

# --- Concurrency and Retry Configuration ---
MAX_WORKERS = 10
MAX_RETRIES = 3
RETRY_DELAY = 5 # in seconds

# =======================================================================
# HELPER FUNCTIONS
# =======================================================================

def clip_score(score):
    """Ensures that a score is within the range 0 to 100."""
    try:
        score = float(score)
        if score < 0:
            return 0
        elif score > 100:
            return 100
        else:
            return score
    except (ValueError, TypeError):
        return None # Return None if score is not a valid number

def parse_filename(filename):
    """Extracts metadata from an inference output filename."""
    # This function should be adapted if your filename conventions change.
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    model = parts[0]
    language = parts[1]
    task = int(re.search(r'task(\d+)', name).group(1))
    setting = "_".join(parts[3:])
    return {"model": model, "language": language, "task": task, "setting": setting}

def load_vlm_responses(filepath):
    """Loads a JSON file of VLM responses."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading VLM responses from {filepath}: {e}")
        return {}

def read_text_file(filepath):
    """Reads content from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file {filepath}: {e}")
        return "Error reading file."

# =======================================================================
# EVALUATION PROMPT GENERATION
# =======================================================================

TASK_INSTRUCTIONS = {
    1: "Analyze the image and list objects prominently visible.",
    # ... (Include all 8 task descriptions as in your original script) ...
    8: "Identify specific named locations mentioned in the text and visually identifiable in the image."
}

def generate_evaluation_prompt(text_id, vlm_response, task_description, language, setting, text_content, image_filename):
    """Generates the full prompt for the LLM-as-a-Judge."""
    prompt = f"""You are a meticulous evaluator of Vision-Language AI responses. Your task is to evaluate the following VLM response.

- **Task**: "{task_description}"
- **Language**: {language}
- **Setting**: {setting.replace('_', ' ')}

The response was generated based on this input:
- **Image File**: {image_filename}
- **Associated Text**: "{text_content}"

**VLM Response to Evaluate (ID: {text_id}):**
"{vlm_response}"

---
**Evaluation Criteria:**
Please rate the quality of this response on a scale from 0 (lowest quality) to 100 (highest quality) based on:
1.  **Accuracy**: Is the response factually correct and logically sound?
2.  **Helpfulness**: Does it directly and clearly address the task?
3.  **Linguistic Quality**: Is the response coherent, natural, and well-written?

**Instructions:**
- Remain objective and do not let response length influence your judgment.
- Be strict in your scoring; a perfect score should be reserved for exceptional responses.
- Your entire output must be ONLY a single, valid JSON object with the score.

**Example Output:**
{{"score": 85}}
"""
    return prompt

# =======================================================================
# LLM-AS-A-JUDGE API INTERACTION
# =======================================================================

def evaluate_one_item(session, text_id, prompt, retry_count=0):
    """Evaluates a single item using the Gemini API with retry logic."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 50, "temperature": 0.0},
        "safetySettings": [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in 
                           ["HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_HATE_SPEECH", 
                            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT"]]
    }
    try:
        response = session.post(GEMINI_ENDPOINT_URL, json=payload, timeout=60)
        response.raise_for_status()
        result_json = response.json()
        
        output_text = result_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        cleaned_text = re.sub(r"```json\s*|\s*```", "", output_text)
        
        try:
            score_data = json.loads(cleaned_text)
            score = clip_score(score_data.get("score"))
            if score is not None:
                return text_id, {"score": score}
        except json.JSONDecodeError:
            print(f"\nWarning: JSON parsing failed for ID {text_id}. Raw text: '{output_text}'")
    
    except requests.exceptions.RequestException as e:
        print(f"\nAPI Error for ID {text_id}: {e}")
    except (KeyError, IndexError) as e:
        print(f"\nUnexpected API response structure for ID {text_id}: {result_json}. Error: {e}")

    if retry_count < MAX_RETRIES:
        time.sleep(RETRY_DELAY)
        return evaluate_one_item(session, text_id, prompt, retry_count + 1)
        
    print(f"\nFailed to evaluate ID {text_id} after {MAX_RETRIES} retries. Assigning default score.")
    return text_id, {"score": 50} # Default score on persistent failure

# =======================================================================
# MAIN EXECUTION LOGIC
# =======================================================================

def main(args):
    """Main function to orchestrate the LLM-as-a-judge evaluation process."""
    if not GEMINI_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set. Please set it before running.")
        return

    print("=====================================================")
    print(" VLURes LLM-as-a-Judge Evaluation Script")
    print("=====================================================")
    print(f"Evaluating outputs for model: {args.model_to_evaluate} in {args.language}")

    # Find all relevant inference output files for the specified model and language
    search_pattern = os.path.join(args.input_dir, f"{args.model_to_evaluate}_{args.language}_*.json")
    files_to_evaluate = glob.glob(search_pattern)

    if not files_to_evaluate:
        print(f"Error: No inference output files found for '{args.model_to_evaluate}' and '{args.language}' in '{args.input_dir}'")
        return

    print(f"Found {len(files_to_evaluate)} files to evaluate.")
    
    for filepath in files_to_evaluate:
        file_meta = parse_filename(filepath)
        print(f"\n--- Processing: {os.path.basename(filepath)} ---")

        vlm_responses = load_vlm_responses(filepath)
        if not vlm_responses:
            print("No responses found. Skipping.")
            continue

        # Prepare paths for this specific evaluation run
        output_filename = os.path.join(args.output_dir, f"scores_{os.path.basename(filepath)}")
        checkpoint_filename = os.path.join(args.checkpoint_dir, f"ckpt_scores_{os.path.basename(filepath)}")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)
        
        # Load checkpoint
        eval_results = OrderedDict()
        if os.path.exists(checkpoint_filename):
            with open(checkpoint_filename, 'r', encoding='utf-8') as f:
                eval_results = OrderedDict(json.load(f))
            print(f"Resuming from checkpoint. {len(eval_results)} items already evaluated.")

        # Prepare tasks for concurrent evaluation
        tasks_to_run = []
        lang_data_path = os.path.join(args.data_dir, file_meta['language'])
        
        for item_id, response_data in vlm_responses.items():
            if str(item_id) in eval_results:
                continue

            text_path = os.path.join(lang_data_path, f"text{item_id}.txt")
            image_filename = f"{item_id}.jpg" # Assumed filename
            
            text_content = read_text_file(text_path)
            if text_content is None:
                continue

            prompt = generate_evaluation_prompt(
                item_id, response_data[f"Task_{file_meta['task']}"], TASK_INSTRUCTIONS[file_meta['task']],
                file_meta['language'], file_meta['setting'], text_content, image_filename
            )
            tasks_to_run.append((item_id, prompt))
            
        if not tasks_to_run:
            print("All items for this file have been evaluated.")
            continue

        # Run evaluation with a thread pool
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_id = {executor.submit(evaluate_one_item, session, item_id, prompt): item_id for item_id, prompt in tasks_to_run}
                
                for future in tqdm(future_to_id.keys(), total=len(tasks_to_run), desc=f"Evaluating Task {file_meta['task']}"):
                    item_id, result = future.result()
                    eval_results[item_id] = result
                    
                    # Save checkpoint frequently
                    if len(eval_results) % 20 == 0:
                        with open(checkpoint_filename, 'w', encoding='utf-8') as f:
                            json.dump(eval_results, f, indent=4)
        
        # Final save for this file
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=4, ensure_ascii=False)
        
        # Remove checkpoint after successful completion
        if os.path.exists(checkpoint_filename):
            os.remove(checkpoint_filename)
        
        print(f"Finished evaluation for {os.path.basename(filepath)}. Results saved.")

    print("\n=====================================================")
    print("All evaluation tasks complete.")
    print("=====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge evaluation for the VLURes benchmark.")
    
    parser.add_argument("--language", type=str, required=True, choices=["English", "Japanese", "Swahili", "Urdu"],
                        help="The language of the outputs to evaluate.")
    parser.add_argument("--model_to_evaluate", type=str, required=True,
                        help="The name of the VLM whose outputs are to be evaluated (e.g., 'GPT-4o').")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INFERENCE_DIR,
                        help="Directory containing the VLM inference outputs.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Root directory with language subfolders containing images/texts.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_EVAL_OUTPUT_DIR,
                        help="Directory to save the final evaluation score files.")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory to save intermediate checkpoints.")
                        
    args = parser.parse_args()
    main(args)

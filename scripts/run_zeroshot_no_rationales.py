import os
import json
import base64
import time
import argparse
from openai import OpenAI
from collections import OrderedDict
from tqdm import tqdm

# =======================================================================
# SCRIPT CONFIGURATION
# =======================================================================

# --- Model and API Configuration ---
# The API key is loaded securely from an environment variable.
# Before running, execute: export OPENAI_API_KEY="your-api-key"
client = OpenAI()

# --- Default File Paths (relative to the repository root) ---
# This structure assumes the script is run from the root of the repo.
# e.g., python scripts/run_zeroshot_no_rationales.py ...
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "results/inference_outputs"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"

# --- Batch Processing and Retry Configuration ---
BATCH_SIZE = 8
MAX_RETRIES = 3
RETRY_DELAY = 5  # in seconds

# =======================================================================
# LANGUAGE AND TASK PROMPT DEFINITIONS
# =======================================================================

# This dictionary holds all prompts and task descriptions.
LANGUAGE_CONFIGS = {
    "English": {
        "code": "En",
        "system_prompt": "You are an AI assistant that analyzes images and text.",
        "prompt_template_image_only": """
            You are an intelligent assistant tasked with analyzing images. Please perform the following task for the given image:
            
            {task_description}
            
            Provide your analysis for this task only, clearly labeled.
            """,
                    "prompt_template_image_text": """
            You are an intelligent assistant tasked with analyzing the relationship between images and text. 
            Please examine both the image and the provided text carefully.
            
            Text associated with the image:
            {text_content}
            
            Task:
            {task_description}
            
            Provide your analysis based on both the image and text. Be specific and reference evidence from both sources.
            """,
        "tasks": {
            # Image-only tasks (1-5)
            1: "Analyze this image and list all objects present. Categorize each object into groups such as furniture, electronic devices, clothing, etc. Be thorough and specific in your identification.",
            2: "Describe the overall scene in this image. What is the setting, and what activities or events are taking place? Provide a comprehensive overview of the environment and any actions occurring.",
            3: "Identify any interactions or relationships between objects or entities in this image. How are they related or interacting with each other? Explain any spatial, functional, or social connections you observe.",
            4: "Divide this image into different semantic regions. Label each region (e.g., sky, buildings, people, street) and briefly describe its contents. Provide a clear breakdown of the image's composition.",
            5: "Provide a detailed, natural language description of what is happening in this image. Narrate the scene as if you were explaining it to someone who cannot see it, including all relevant details and actions.",
            
            # Image-text tasks (6-8)
            6: "Extract and list the specific parts of the text that closely match or directly reference entities, objects, or scenes depicted in the image. Be precise in identifying these connections and explain the visual evidence that supports each textual reference.",
            7: "Identify which parts of the text are not relevant to or not represented in the image. Explain why these elements are unrelated by describing what is missing in the image that would be needed to illustrate these textual elements.",
            8: "What places are mentioned in the text or shown in the image? For each place identified, indicate whether it appears in the text, the image, or both. If any of these places are famous or well-known locations, explain why they are significant."
        }
    },
    "Japanese": {
        "code": "Jp",
        "system_prompt": "あなたは画像とテキストを分析し、日本語で回答する AI アシスタントです。",
        "prompt_template_image_only": """
              あなたは画像を分析し、日本語で回答する知的なアシスタントです。
              以下のタスクに従って、与えられた画像を分析してください：
              
              {task_description}
              
              このタスクに対する分析のみを明確にラベル付けして日本語で提供してください。
              """,
                      "prompt_template_image_text": """
              あなたは画像とテキストの関係を分析し、日本語で回答する知的なアシスタントです。
              画像と提供されたテキストの両方を注意深く検討してください。
              
              画像に関連するテキスト:
              {text_content}
              
              タスク:
              {task_description}
              
              画像とテキストの両方に基づいてあなたの分析を提供してください。具体的で、両方のソースからの証拠を参照してください。
              """,
        "tasks": {
            # Image-only tasks (1-5)
            1: "この画像に存在するすべてのオブジェクトを分析し、リストアップしてください。家具、電子機器、衣類などのグループにオブジェクトを分類してください。識別は徹底的かつ具体的に行ってください。",
            2: "この画像の全体的な場面を説明してください。どのような設定で、どのような活動や出来事が起こっているでしょうか？環境や発生している行動の包括的な概要を提供してください。",
            3: "この画像内のオブジェクトや実体間の相互作用や関係を特定してください。それらはどのように関連し、相互作用していますか？空間的、機能的、または社会的なつながりを説明してください。",
            4: "この画像を異なる意味領域に分割してください。各領域（例：空、建物、人、通りなど）にラベルを付け、その内容を簡潔に説明してください。画像の構成を明確に分類してください。",
            5: "この画像で起こっていることの詳細な自然言語による説明を提供してください。まるで見ることができない人に説明するかのように、すべての関連する詳細や行動を含めて場面を語ってください。",
            
            # Image-text tasks (6-8)
            6: "テキストの特定の部分で、画像に描かれているエンティティ、オブジェクト、またはシーンに密接に一致または直接言及している部分を抽出してリストアップしてください。これらの接続を特定する際に正確であり、各テキスト参照をサポートする視覚的証拠を説明してください。",
            7: "テキストのどの部分が画像に関連していないか、または画像に表現されていないかを特定してください。これらのテキスト要素を説明するために画像に必要なものが何が欠けているかを説明して、これらの要素が無関係である理由を説明してください。",
            8: "テキストや画像で言及されている場所はどこですか？特定された各場所について、それがテキスト、画像、またはその両方に現れるかを示してください。これらの場所のいずれかが有名または広く知られている場所である場合、それらが重要である理由を説明してください。"
        }
    },
    "Swahili": {
        "code": "Sw",
        "system_prompt": "Wewe ni AI msaidizi unayechambua picha na maandishi na kutoa majibu kwa lugha ya Kiswahili.",
        "prompt_template_image_only": """
              Wewe ni msaidizi wa akili unaechambua picha na kutoa majibu kwa lugha ya Kiswahili.
              Tafadhali chambua picha uliyopewa kwa kufuata maelekezo yafuatayo:
              
              {task_description}
              
              Tafadhali toa uchambuzi wako kwa lugha ya Kiswahili pekee, na uweke lebo wazi.
              """,
                      "prompt_template_image_text": """
              Wewe ni msaidizi wa akili unayechambua uhusiano kati ya picha na maandishi na kutoa majibu kwa lugha ya Kiswahili.
              Tafadhali chunguza kwa makini picha na maandishi yaliyotolewa.
              
              Maandishi yanayohusiana na picha:
              {text_content}
              
              Kazi:
              {task_description}
              
              Toa uchambuzi wako ukizingatia picha na maandishi. Kuwa mahususi na taja ushahidi kutoka vyanzo vyote viwili.
              """,
        "tasks": {
            # Image-only tasks (1-5)
            1: "Changanua picha hii na uorodheshe vitu vyote vilivyomo. Ainisha kila kitu katika makundi kama vile samani, vifaa vya elektroniki, mavazi, n.k. Kuwa makini na maalum katika utambulisho wako.",
            2: "Elezea mandhari nzima katika picha hii. Mazingira ni yapi, na ni shughuli au matukio gani yanayoendelea? Toa maelezo ya kina ya mazingira na vitendo vyovyote vinavyofanyika.",
            3: "Tambua mahusiano yoyote au uhusiano kati ya vitu au viumbe katika picha hii. Vinahusianaje au vinaathirianaje? Eleza uhusiano wowote wa kimwili, kiutendaji, au kijamii unaoona.",
            4: "Gawanya picha hii katika maeneo tofauti ya kisemantiki. Taja kila eneo (k.m. anga, majengo, watu, barabara) na uelezee kwa ufupi yaliyomo. Toa mgawanyo wazi wa muundo wa picha.",
            5: "Toa maelezo ya kina ya kimaandishi kuhusu kinachotokea katika picha hii. Simulia tukio kana kwamba unamwelezea mtu asiyeweza kuiona, ukijumuisha maelezo yote muhimu na vitendo.",
            
            # Image-text tasks (6-8)
            6: "Toa na uorodheshe sehemu mahususi za maandishi zinazofanana sana au zinazorejelea moja kwa moja vitu, viumbe, au mandhari zilizooneshwa katika picha. Kuwa sahihi katika kutambua uhusiano huu na ueleze ushahidi wa kuona unaounga mkono kila rejeleo la maandishi.",
            7: "Tambua ni sehemu gani za maandishi ambazo hazihusiani au haziwakilishwi katika picha. Eleza kwa nini vipengele hivi havihusiani kwa kuelezea kile kinachokosekana katika picha ambacho kingekuwa muhimu kufafanua vipengele hivi vya maandishi.",
            8: "Ni maeneo gani yametajwa katika maandishi au yanaonekana katika picha? Kwa kila eneo lililotambuliwa, onyesha kama linaonekana katika maandishi, picha, au vyote. Ikiwa maeneo haya yoyote ni mashuhuri au yanayojulikana sana, eleza kwa nini yana umuhimu."
        }
    },
    "Urdu": {
        "code": "Ur",
        "system_prompt": "آپ ایک ایسے AI اسسٹنٹ ہیں جو تصاویر اور متن کا تجزیہ کرتے ہیں اور اردو میں جوابات فراہم کرتے ہیں۔",
        "prompt_template_image_only": """
          آپ ایک ایسے ذہین اسسٹنٹ ہیں جو تصاویر کا تجزیہ کرتے ہیں اور اردو میں جوابات فراہم کرتے ہیں۔
          براہ کرم درج ذیل ٹاسک کے مطابق، دی گئی تصویر کا تجزیہ کریں:
          
          {task_description}
          
          براہ کرم صرف اس ٹاسک کے لیے اپنا تجزیہ واضح طور پر اردو میں پیش کریں۔
          """,
                  "prompt_template_image_text": """
          آپ ایک ایسے ذہین اسسٹنٹ ہیں جو تصاویر اور متن کے درمیان تعلق کا تجزیہ کرتے ہیں اور اردو میں جوابات فراہم کرتے ہیں۔
          براہ کرم تصویر اور فراہم کردہ متن دونوں کا احتیاط سے جائزہ لیں۔
          
          تصویر سے متعلق متن:
          {text_content}
          
          ٹاسک:
          {task_description}
          
          تصویر اور متن دونوں کے مطابق اپنا تجزیہ فراہم کریں۔ مخصوص ہوں اور دونوں ذرائع سے شواہد کا حوالہ دیں۔
          """,
        "tasks": {
            # Image-only tasks (1-5)
            1: "اس تصویر کا تجزیہ کریں اور موجود تمام اشیاء کی فہرست بنائیں۔ ہر شے کو گروپس میں درجہ بندی کریں جیسے فرنیچر، الیکٹرانک آلات، کپڑے، وغیرہ۔ اپنی شناخت میں جامع اور مخصوص رہیں۔",
            2: "اس تصویر میں مجموعی منظر کی تفصیل بیان کریں۔ ماحول کیا ہے، اور کون سی سرگرمیاں یا واقعات پیش آرہے ہیں؟ ماحول اور کسی بھی قابل ذکر کارروائی کا جامع جائزہ فراہم کریں۔",
            3: "اس تصویر میں اشیاء یا اکائیوں کے درمیان کسی بھی تعامل یا تعلقات کی نشاندہی کریں۔ وہ ایک دوسرے سے کیسے متعلق ہیں یا تعامل کر رہے ہیں؟ کسی بھی مکانی، فعال، یا سماجی روابط کی وضاحت کریں جو آپ دیکھتے ہیں۔",
            4: "اس تصویر کو مختلف معنی خیز علاقوں میں تقسیم کریں۔ ہر علاقے کو لیبل کریں (مثلاً، آسمان، عمارات، لوگ، سڑک) اور مختصراً اس کے مواد کی وضاحت کریں۔ تصویر کی ساخت کا واضح تجزیہ فراہم کریں۔",
            5: "اس تصویر میں کیا ہو رہا ہے اس کی تفصیلی، قدرتی زبان میں وضاحت فراہم کریں۔ منظر کی ایسے بیان کریں جیسے آپ کسی ایسے شخص کو سمجھا رہے ہوں جو اسے دیکھ نہیں سکتا، تمام متعلقہ تفصیلات اور کارروائیوں کو شامل کرتے ہوئے۔",
            
            # Image-text tasks (6-8)
            6: "متن کے مخصوص حصے نکالیں اور فہرست بنائیں جو تصویر میں دکھائے گئے اداروں، اشیاء، یا مناظر سے قریبی مماثلت رکھتے ہیں یا براہ راست ان کا حوالہ دیتے ہیں۔ ان روابط کی شناخت میں درست رہیں اور ہر متنی حوالے کی تائید کرنے والے بصری شواہد کی وضاحت کریں۔",
            7: "متن کے کون سے حصے تصویر سے متعلق نہیں ہیں یا تصویر میں نمائندگی نہیں کرتے ہیں، اس کی نشاندہی کریں۔ ان عناصر کے غیر متعلقہ ہونے کی وجہ بیان کریں، یہ وضاحت کرکے کہ تصویر میں ان متنی عناصر کی وضاحت کے لیے کیا چیز غائب ہے۔",
            8: "متن یا تصویر میں کون سی جگہوں کا ذکر کیا گیا ہے؟ شناخت شدہ ہر جگہ کے لیے، بتائیں کہ یہ متن میں، تصویر میں، یا دونوں میں ظاہر ہوتی ہے۔ اگر ان میں سے کوئی بھی جگہ مشہور یا جانی پہچانی مقامات ہیں، تو بتائیں کہ وہ کیوں اہم ہیں۔"
        }
    }
}

# =======================================================================
# HELPER FUNCTIONS
# =======================================================================

def encode_image(image_path):
    """Encodes an image to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def read_text_file(text_path):
    """Reads content from a text file."""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file {text_path}: {e}")
        return None

def get_image_id(filename):
    """Extracts a numeric ID from a filename."""
    return ''.join(filter(str.isdigit, os.path.splitext(filename)[0]))

# =======================================================================
# API INTERACTION LOGIC
# =======================================================================

def process_batch(client, model, messages, max_tokens, temperature, retry_count=0):
    """Sends a single batch request to the OpenAI API with retry logic."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error processing batch: {e}")
        if retry_count < MAX_RETRIES:
            print(f"Retrying in {RETRY_DELAY} seconds... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
            return process_batch(client, model, messages, max_tokens, temperature, retry_count + 1)
        else:
            print("Max retries reached. Skipping this batch.")
            return None

# =======================================================================
# MAIN SCRIPT LOGIC
# =======================================================================

def main(args):
    """Main function to run the inference process."""
    
    # --- Setup and Configuration ---
    lang_config = LANGUAGE_CONFIGS.get(args.language)
    if not lang_config:
        print(f"Error: Language '{args.language}' not configured.")
        return

    task_description = lang_config["tasks"].get(args.task)
    if not task_description:
        print(f"Error: Task number '{args.task}' not found for language '{args.language}'.")
        return
        
    print("=====================================================")
    print(f"Starting VLM Inference: Zero-Shot without Rationales")
    print("=====================================================")
    print(f"  Model:      {args.model}")
    print(f"  Language:   {args.language}")
    print(f"  Task:       {args.task} - {task_description[:50]}...")
    
    # --- Prepare File Paths and Checkpoints ---
    data_path = os.path.join(args.data_dir, lang_config['code'])
    output_filename = os.path.join(args.output_dir, f"{args.model}_{args.language}_task{args.task}_zeroshot_no_rationales.json")
    checkpoint_filename = os.path.join(args.checkpoint_dir, f"ckpt_{args.model}_{args.language}_task{args.task}_zeroshot_no_rationales.json")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Load Data and Checkpoints ---
    all_files = sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=get_image_id)
    
    results = OrderedDict()
    if os.path.exists(checkpoint_filename):
        with open(checkpoint_filename, 'r', encoding='utf-8') as f:
            results = OrderedDict(json.load(f))
        print(f"\nResuming from checkpoint. {len(results)} items already processed.")
    
    processed_ids = {str(k) for k in results.keys()}
    files_to_process = [f for f in all_files if get_image_id(f) not in processed_ids]
    
    if not files_to_process:
        print("All items have already been processed. Exiting.")
        return

    # --- Determine Task Type and Process Data ---
    is_image_text_task = args.task >= 6
    
    if is_image_text_task:
        print(f"\nProcessing {len(files_to_process)} image-text pairs...")
        prompt_template = lang_config["prompt_template_image_text"]
        
        for filename in tqdm(files_to_process, desc="Processing Image-Text Pairs"):
            image_id = get_image_id(filename)
            image_path = os.path.join(data_path, filename)
            text_path = os.path.join(data_path, f"text{image_id}.txt")
            
            if not os.path.exists(text_path):
                print(f"Warning: Text file not found for image {filename}. Skipping.")
                continue

            base64_image = encode_image(image_path)
            text_content = read_text_file(text_path)
            
            if not base64_image or text_content is None:
                continue

            # Format the prompt for this specific pair
            formatted_prompt = prompt_template.format(
                task_description=task_description,
                text_content=text_content
            )

            messages = [
                {"role": "system", "content": lang_config["system_prompt"]},
                {"role": "user", "content": [
                    {"type": "text", "text": formatted_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
            
            result = process_batch(client, args.model, messages, args.max_tokens, args.temperature)
            if result:
                results[image_id] = result
                
            # Save checkpoint after each processed item
            with open(checkpoint_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    else: # Image-only tasks
        print(f"\nProcessing {len(files_to_process)} images in batches of {BATCH_SIZE}...")
        prompt = lang_config["prompt_template_image_only"].format(task_description=task_description)
        
        for i in tqdm(range(0, len(files_to_process), BATCH_SIZE), desc="Processing Image Batches"):
            batch_filenames = files_to_process[i:i+BATCH_SIZE]
            batch_image_paths = [os.path.join(data_path, f) for f in batch_filenames]
            
            content = [{"type": "text", "text": prompt}]
            for img_path in batch_image_paths:
                base64_image = encode_image(img_path)
                if base64_image:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

            if len(content) == 1: # No valid images in batch
                continue

            messages = [
                {"role": "system", "content": lang_config["system_prompt"]},
                {"role": "user", "content": content}
            ]
            
            result = process_batch(client, args.model, messages, args.max_tokens, args.temperature)
            
            if result:
                # Naively assign the single response to all images in the batch
                # Note: The Vision API may not provide per-image responses in a multi-image prompt.
                # This assumes a single, consolidated analysis is acceptable.
                for filename in batch_filenames:
                    image_id = get_image_id(filename)
                    results[image_id] = result

            # Save checkpoint after each processed batch
            with open(checkpoint_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # --- Final Save ---
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\nProcessing complete. Final results saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM inference for the VLURes benchmark.")
    
    parser.add_argument("--language", type=str, required=True, choices=LANGUAGE_CONFIGS.keys(),
                        help="The language to process.")
    parser.add_argument("--task", type=int, required=True, choices=range(1, 9),
                        help="The task number to run (1-8).")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="The model name to use (e.g., 'gpt-4o').")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Root directory containing the language-specific data folders.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save the final JSON output.")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory to save intermediate checkpoints.")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum number of tokens for the model to generate.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for the model.")
                        
    args = parser.parse_args()
    main(args)

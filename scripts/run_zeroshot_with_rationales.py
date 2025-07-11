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
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "results/inference_outputs"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"

# --- Batch Processing and Retry Configuration ---
BATCH_SIZE = 8
MAX_RETRIES = 3
RETRY_DELAY = 5  # in seconds

# =======================================================================
# LANGUAGE AND TASK PROMPT DEFINITIONS (WITH RATIONALES)
# =======================================================================

# This dictionary holds all prompts and task descriptions with Chain-of-Thought style rationales.
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
                                # Image-only tasks (1-5) with Chain of Thought
                                1: """Let's analyze this image step by step to identify and categorize all objects present.

                    Steps to follow:
                    1. First, scan the entire image systematically from left to right, top to bottom
                    2. List each object you identify
                    3. Group similar objects into categories (furniture, electronics, clothing, etc.)
                    4. Double-check for any small or partially visible objects
                    5. Ensure no objects are missed in any part of the image

                    After following these steps, provide:
                    1. Your detailed response listing and categorizing all objects
                    2. Your rationale: Explain how you identified these objects and why you categorized them as you did

                    Think step by step
                    """,
                                2: """Let's analyze this scene step by step to provide a comprehensive description.

                    Steps to follow:
                    1. Identify the primary setting/location
                    2. Note the time of day and general atmosphere
                    3. Identify main activities or events occurring
                    4. Observe any secondary or background activities
                    5. Consider the overall context and purpose of the scene

                    After following these steps, provide:
                    1. Your detailed description of the overall scene
                    2. Your rationale: Explain how you determined the setting and activities, and what visual cues led to your interpretation

                    Think step by step
                    """,
                                3: """Let's analyze the interactions and relationships in this image step by step.

                    Steps to follow:
                    1. Identify all entities (objects/people) in the image
                    2. Examine spatial relationships between entities
                    3. Observe any direct interactions
                    4. Consider functional relationships
                    5. Analyze any implied or social connections

                    After following these steps, provide:
                    1. Your detailed analysis of all interactions and relationships
                    2. Your rationale: Explain how you identified these relationships and what evidence supports your conclusions

                    Think step by step
                    """,
                                4: """Let's divide this image into semantic regions step by step.

                    Steps to follow:
                    1. Identify major spatial divisions in the image
                    2. Determine the function or nature of each region
                    3. Note the contents within each region
                    4. Observe how regions connect or transition
                    5. Consider the hierarchical importance of each region

                    After following these steps, provide:
                    1. Your detailed breakdown of the image's semantic regions
                    2. Your rationale: Explain how you determined these regions and why you structured them this way

                    Think step by step
                    """,
                                5: """Let's create a detailed narrative description of this image step by step.

                    Steps to follow:
                    1. Establish the main subject or focus
                    2. Describe the immediate context and surroundings
                    3. Note any actions or movements
                    4. Include relevant details about appearance and condition
                    5. Consider the temporal aspects (what led to this moment)

                    After following these steps, provide:
                    1. Your detailed narrative description
                    2. Your rationale: Explain how you constructed this narrative and what key elements informed your description

                    Think step by step
                    """,
                                
                                # Image-text tasks (6-8) with Chain of Thought
                                6: """Let's analyze the relationship between the image and text step by step to identify matching elements.

                    Steps to follow:
                    1. First, carefully observe all visual elements in the image
                    2. Read the text thoroughly to understand what it describes
                    3. Create a mental list of key entities/concepts mentioned in the text
                    4. For each entity in the text, search for its visual representation in the image
                    5. Note which textual elements have clear visual correspondences
                    6. Pay attention to details like attributes, actions, and spatial relationships

                    After following these steps, provide:
                    1. Your detailed analysis of which specific parts of the text match elements in the image
                    2. Your rationale: Explain exactly how each identified text portion corresponds to visual elements, citing specific visual evidence

                    Think step by step
                    """,
                                7: """Let's identify which parts of the text are unrelated to the image step by step.

                    Steps to follow:
                    1. First, thoroughly document all visual elements present in the image
                    2. Carefully read the text and identify all distinct statements or claims
                    3. For each textual element, actively search for corresponding visual evidence
                    4. Create two lists: text elements with visual support and text elements without visual support
                    5. For elements without visual support, double-check the image to confirm they're truly absent
                    6. Consider whether any text elements might be implied but not directly visible

                    After following these steps, provide:
                    1. Your detailed analysis of which specific parts of the text are not related to the image
                    2. Your rationale: Explain why these elements are unrelated by describing what would need to be present in the image to support them

                    Think step by step
                    """,
                                8: """Let's identify and analyze places mentioned in the image and text step by step.

                    Steps to follow:
                    1. First, identify any locations/places visible in the image
                    2. Search the text for explicit mentions of any places or locations
                    3. For each place identified, determine if it appears in the image, the text, or both
                    4. For places that might be famous, assess whether they have cultural, historical, or general significance
                    5. Gather evidence that supports the identification of each place
                    6. Consider the context to determine why these places are relevant to the content

                    After following these steps, provide:
                    1. Your detailed analysis of all places mentioned in either the image or text
                    2. For each place, specify whether it appears in the image, text, or both
                    3. For famous places, explain their significance
                    4. Your rationale: Explain how you identified each place and what evidence supports your conclusions

                    Think step by step
                    """
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
                            # Image-only tasks (1-5) with Chain of Thought
                            1: """この画像に存在するすべてのオブジェクトを段階的に分析しましょう。

                手順：
                1. まず、画像全体を左から右へ、上から下へと体系的にスキャンします
                2. 識別した各オブジェクトをリストアップします
                3. 類似したオブジェクトをグループ（家具、電子機器、衣類など）に分類します
                4. 小さなオブジェクトや部分的に見えるオブジェクトがないか再確認します
                5. 画像のどの部分にもオブジェクトが見落とされていないことを確認します

                これらの手順に従った後、提供してください：
                1. すべてのオブジェクトをリストアップし分類した詳細な回答
                2. あなたの根拠：これらのオブジェクトをどのように識別し、なぜそのように分類したかを説明してください

                段階的に考えてください
                """,
                            2: """この場面を段階的に分析して、包括的な説明を提供しましょう。

                手順：
                1. 主な設定/場所を特定します
                2. 時間帯と全体的な雰囲気を記録します
                3. 行われている主な活動や出来事を特定します
                4. 二次的または背景の活動を観察します
                5. シーンの全体的な文脈と目的を考慮します

                これらの手順に従った後、提供してください：
                1. 全体的な場面の詳細な説明
                2. あなたの根拠：設定と活動をどのように判断したか、どの視覚的手がかりがあなたの解釈につながったかを説明してください

                段階的に考えてください
                """,
                            3: """この画像の相互作用と関係を段階的に分析しましょう。

                手順：
                1. 画像内のすべてのエンティティ（オブジェクト/人）を特定します
                2. エンティティ間の空間的関係を調べます
                3. 直接的な相互作用を観察します
                4. 機能的な関係を考慮します
                5. 暗示的または社会的なつながりを分析します

                これらの手順に従った後、提供してください：
                1. すべての相互作用と関係の詳細な分析
                2. あなたの根拠：これらの関係をどのように特定し、どの証拠があなたの結論をサポートしているかを説明してください

                段階的に考えてください
                """,
                            4: """この画像を意味のある領域に段階的に分割しましょう。

                手順：
                1. 画像の主要な空間的区分を特定します
                2. 各領域の機能または性質を決定します
                3. 各領域内の内容を記録します
                4. 領域がどのように接続または移行するかを観察します
                5. 各領域の階層的重要性を考慮します

                これらの手順に従った後、提供してください：
                1. 画像の意味的領域の詳細な内訳
                2. あなたの根拠：これらの領域をどのように決定し、なぜこのように構成したかを説明してください

                段階的に考えてください
                """,
                            5: """この画像の詳細な物語的説明を段階的に作成しましょう。

                手順：
                1. 主な被写体または焦点を確立します
                2. 直接の文脈と周囲を説明します
                3. 行動や動きに注目します
                4. 外観や状態に関する関連詳細を含めます
                5. 時間的側面（この瞬間に至った経緯）を考慮します

                これらの手順に従った後、提供してください：
                1. あなたの詳細な物語的説明
                2. あなたの根拠：この物語をどのように構築し、どの主要要素があなたの説明に影響を与えたかを説明してください

                段階的に考えてください
                """,
                            
                            # Image-text tasks (6-8) with Chain of Thought
                            6: """画像とテキストの関係を段階的に分析して、一致する要素を特定しましょう。

                手順：
                1. まず、画像内のすべての視覚的要素を注意深く観察します
                2. テキストを徹底的に読み、何が説明されているかを理解します
                3. テキストで言及されている主要なエンティティ/概念のメンタルリストを作成します
                4. テキスト内の各エンティティについて、画像内のその視覚的表現を探します
                5. どのテキスト要素が明確な視覚的対応を持っているかを記録します
                6. 属性、行動、空間的関係などの詳細に注意を払います

                これらの手順に従った後、提供してください：
                1. テキストのどの特定の部分が画像の要素と一致するかの詳細な分析
                2. あなたの根拠：特定された各テキスト部分が視覚的要素にどのように対応するか、具体的な視覚的証拠を挙げて説明してください

                段階的に考えてください
                """,
                            7: """テキストのどの部分が画像と関連していないかを段階的に特定しましょう。

                手順：
                1. まず、画像に存在するすべての視覚的要素を徹底的に文書化します
                2. テキストを注意深く読み、すべての異なるステートメントまたは主張を特定します
                3. 各テキスト要素について、対応する視覚的証拠を積極的に探します
                4. 2つのリストを作成します：視覚的サポートがあるテキスト要素と視覚的サポートがないテキスト要素
                5. 視覚的サポートがない要素については、それらが本当に不在であることを確認するために画像を再確認します
                6. テキスト要素が暗示されているが直接視覚的に見えない可能性を考慮します

                これらの手順に従った後、提供してください：
                1. テキストのどの特定の部分が画像と関連していないかの詳細な分析
                2. あなたの根拠：これらの要素がそれらをサポートするために画像に存在する必要があるものを説明することによって、なぜこれらの要素が関連していないかを説明してください

                段階的に考えてください
                """,
                            8: """画像とテキストで言及されている場所を段階的に特定し分析しましょう。

                手順：
                1. まず、画像に見える場所/ロケーションを特定します
                2. テキスト内で場所や位置の明示的な言及を探します
                3. 特定された各場所について、それが画像、テキスト、またはその両方に現れるかどうかを判断します
                4. 有名かもしれない場所については、それらが文化的、歴史的、または一般的な重要性を持つかどうかを評価します
                5. 各場所の識別をサポートする証拠を収集します
                6. これらの場所がコンテンツにとって関連性があるのはなぜかを判断するためにコンテキストを考慮します

                これらの手順に従った後、提供してください：
                1. 画像またはテキストのいずれかで言及されているすべての場所の詳細な分析
                2. 各場所について、それが画像、テキスト、またはその両方に現れるかを指定します
                3. 有名な場所については、その重要性を説明します
                4. あなたの根拠：各場所をどのように特定し、どの証拠があなたの結論をサポートしているかを説明してください

                段階的に考えてください
                """
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
                            # Image-only tasks (1-5) with Chain of Thought
                            1: """Hebu tuchambuwe picha hii hatua kwa hatua ili kutambua na kupanga vitu vyote vilivyomo.

                Hatua za kufuata:
                1. Kwanza, angalia picha nzima kwa utaratibu kutoka kushoto kwenda kulia, juu hadi chini
                2. Orodhesha kila kitu unachokitambua
                3. Panga vitu vinavyofanana katika makundi (samani, vifaa vya elektroniki, mavazi, n.k.)
                4. Hakikisha vitu vidogo au vinavyoonekana kwa sehemu
                5. Hakikisha hakuna vitu vilivyosahaulika katika sehemu yoyote ya picha

                Baada ya kufuata hatua hizi, toa:
                1. Jibu lako la kina linaloorodhesha na kupanga vitu vyote
                2. Maelezo yako: Eleza jinsi ulivyotambua vitu hivi na kwa nini ulivivipanga kama ulivyofanya

                Fikiria hatua kwa hatua
                """,
                            2: """Hebu tuchambuwe mandhari hii hatua kwa hatua ili kutoa maelezo kamili.

                Hatua za kufuata:
                1. Tambua eneo kuu/mahali
                2. Angalia wakati wa siku na hali ya jumla ya hewa
                3. Tambua shughuli au matukio makuu yanayoendelea
                4. Angalia shughuli zozote za ziada au za nyuma
                5. Fikiria muktadha wa jumla na madhumuni ya tukio

                Baada ya kufuata hatua hizi, toa:
                1. Maelezo yako ya kina ya mandhari nzima
                2. Maelezo yako: Eleza jinsi ulivyoamua mazingira na shughuli, na ni ishara zipi za kuona zilizoongoza tafsiri yako

                Fikiria hatua kwa hatua
                """,
                            3: """Hebu tuchambuwe mwingiliano na uhusiano katika picha hii hatua kwa hatua.

                Hatua za kufuata:
                1. Tambua vitu vyote (vitu/watu) katika picha
                2. Chunguza uhusiano wa nafasi kati ya vitu
                3. Angalia mwingiliano wowote wa moja kwa moja
                4. Fikiria uhusiano wa kimajukumu
                5. Chambua uhusiano wowote uliodokezwa au wa kijamii

                Baada ya kufuata hatua hizi, toa:
                1. Uchambuzi wako wa kina wa mwingiliano na uhusiano wote
                2. Maelezo yako: Eleza jinsi ulivyotambua uhusiano huu na ni ushahidi gani unaounga mkono hitimisho lako

                Fikiria hatua kwa hatua
                """,
                            4: """Hebu tugawanye picha hii katika maeneo ya kimaana hatua kwa hatua.

                Hatua za kufuata:
                1. Tambua mgawanyiko mkuu wa nafasi katika picha
                2. Amua kazi au asili ya kila eneo
                3. Angalia yaliyomo ndani ya kila eneo
                4. Angalia jinsi maeneo yanavyounganishwa au kubadilishana
                5. Fikiria umuhimu wa ngazi ya kila eneo

                Baada ya kufuata hatua hizi, toa:
                1. Uchambuzi wako wa kina wa maeneo ya kimaana ya picha
                2. Maelezo yako: Eleza jinsi ulivyoamua maeneo haya na kwa nini uliyapanga hivi

                Fikiria hatua kwa hatua
                """,
                            5: """Hebu tuunde maelezo ya kina ya masimulizi ya picha hii hatua kwa hatua.

                Hatua za kufuata:
                1. Anzisha mada kuu au lengo
                2. Eleza muktadha wa papo hapo na mazingira
                3. Angalia vitendo au harakati zozote
                4. Jumuisha maelezo muhimu kuhusu mwonekano na hali
                5. Fikiria vipengele vya muda (kilichosababisha wakati huu)

                Baada ya kufuata hatua hizi, toa:
                1. Maelezo yako ya kina ya masimulizi
                2. Maelezo yako: Eleza jinsi ulivyounda masimulizi haya na ni vipengele gani muhimu vilivyoathiri maelezo yako

                Fikiria hatua kwa hatua
                """,
                            
                            # Image-text tasks (6-8) with Chain of Thought
                            6: """Hebu tuchambuwe uhusiano kati ya picha na maandishi hatua kwa hatua ili kutambua vipengele vinavyolingana.

                Hatua za kufuata:
                1. Kwanza, angalia kwa makini vipengele vyote vya kuona katika picha
                2. Soma maandishi kwa undani ili kuelewa yanayoelezea
                3. Unda orodha ya kiakili ya vitu/dhana muhimu zilizotajwa katika maandishi
                4. Kwa kila kitu kilichotajwa katika maandishi, tafuta uwakilishi wake wa kuona katika picha
                5. Angalia ni vipengele vipi vya maandishi vina uhusiano wa wazi wa kuona
                6. Zingatie maelezo kama vile sifa, vitendo na uhusiano wa nafasi

                Baada ya kufuata hatua hizi, toa:
                1. Uchambuzi wako wa kina wa sehemu gani mahususi za maandishi zinazolingana na vipengele katika picha
                2. Maelezo yako: Eleza kwa usahihi jinsi kila sehemu ya maandishi iliyotambuliwa inavyohusiana na vipengele vya kuona, ukitaja ushahidi mahususi wa kuona

                Fikiria hatua kwa hatua
                """,
                            7: """Hebu tutambue ni sehemu gani za maandishi hazihusiani na picha hatua kwa hatua.

                Hatua za kufuata:
                1. Kwanza, andika kwa undani vipengele vyote vya kuona vilivyo katika picha
                2. Soma kwa makini maandishi na utambue taarifa au madai yote tofauti
                3. Kwa kila kipengele cha maandishi, tafuta kwa bidii ushahidi unaolingana wa kuona
                4. Unda orodha mbili: vipengele vya maandishi vilivyo na msaada wa kuona na vipengele vya maandishi visivyo na msaada wa kuona
                5. Kwa vipengele visivyo na msaada wa kuona, angalia tena picha ili kuthibitisha kwamba havipo kweli
                6. Fikiria kama vipengele vyovyote vya maandishi vinaweza kudokezwa lakini havionekani moja kwa moja

                Baada ya kufuata hatua hizi, toa:
                1. Uchambuzi wako wa kina wa ni sehemu gani mahususi za maandishi hazihusiani na picha
                2. Maelezo yako: Eleza kwa nini vipengele hivi havihusiani kwa kuelezea nini kinahitajika kuwepo katika picha ili kuviunga mkono

                Fikiria hatua kwa hatua
                """,
                            8: """Hebu tutambue na kuchambua maeneo yaliyotajwa katika picha na maandishi hatua kwa hatua.

                Hatua za kufuata:
                1. Kwanza, tambua maeneo/mahali popote panapoonekana katika picha
                2. Tafuta katika maandishi matajo ya wazi ya maeneo yoyote au mahali
                3. Kwa kila eneo lililotambuliwa, amua kama linaonekana katika picha, maandishi, au vyote
                4. Kwa maeneo yanayoweza kuwa maarufu, tathmini kama yana umuhimu wa kitamaduni, kihistoria, au kwa ujumla
                5. Kusanya ushahidi unaoiunga mkono utambulisho wa kila eneo
                6. Fikiria muktadha ili kuamua kwa nini maeneo haya yanahusiana na maudhui

                Baada ya kufuata hatua hizi, toa:
                1. Uchambuzi wako wa kina wa maeneo yote yaliyotajwa katika picha au maandishi
                2. Kwa kila eneo, fafanua kama linaonekana katika picha, maandishi, au vyote
                3. Kwa maeneo maarufu, eleza umuhimu wake
                4. Maelezo yako: Eleza jinsi ulivyotambua kila eneo na ni ushahidi gani unaunga mkono hitimisho lako

                Fikiria hatua kwa hatua
                """
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
                                    # Image-only tasks (1-5) with Chain of Thought
                                    1: """آئیے اس تصویر کا مرحلہ وار تجزیہ کریں تاکہ موجود تمام اشیاء کی شناخت اور زمرہ بندی کی جا سکے۔

                        پیروی کرنے کے لیے مراحل:
                        1. سب سے پہلے، پوری تصویر کو باقاعدگی سے بائیں سے دائیں، اوپر سے نیچے تک اسکین کریں
                        2. ہر شے کی فہرست بنائیں جسے آپ شناخت کرتے ہیں
                        3. ملتی جلتی اشیاء کو گروپس میں درجہ بندی کریں (فرنیچر، الیکٹرانک آلات، کپڑے، وغیرہ)
                        4. چھوٹی یا جزوی طور پر نظر آنے والی اشیاء کی دوبارہ جانچ کریں
                        5. اس بات کو یقینی بنائیں کہ تصویر کے کسی بھی حصے میں کوئی شے نظر انداز نہ ہو

                        ان مراحل کی پیروی کے بعد، فراہم کریں:
                        1. تمام اشیاء کی فہرست اور زمرہ بندی کرتے ہوئے آپ کا تفصیلی جواب
                        2. آپ کی دلیل: وضاحت کریں کہ آپ نے ان اشیاء کی شناخت کیسے کی اور آپ نے انھیں اس طرح کیوں درجہ بندی کیا

                        مرحلہ وار سوچیں
                        """,
                                    2: """آئیے اس منظر کا مرحلہ وار تجزیہ کریں تاکہ ایک جامع تفصیل فراہم کی جا سکے۔

                        پیروی کرنے کے لیے مراحل:
                        1. بنیادی سیٹنگ/مقام کی شناخت کریں
                        2. دن کے وقت اور عمومی ماحول کا ذکر کریں
                        3. ہونے والی اہم سرگرمیوں یا واقعات کی شناخت کریں
                        4. کسی بھی ثانوی یا پس منظر کی سرگرمیوں کا مشاہدہ کریں
                        5. منظر کے مجموعی تناظر اور مقصد پر غور کریں

                        ان مراحل کی پیروی کے بعد، فراہم کریں:
                        1. مجموعی منظر کی آپ کی تفصیلی تفصیل
                        2. آپ کی دلیل: وضاحت کریں کہ آپ نے سیٹنگ اور سرگرمیوں کا تعین کیسے کیا، اور کون سے بصری اشارے آپ کی تشریح کی طرف لے گئے

                        مرحلہ وار سوچیں
                        """,
                                    3: """آئیے اس تصویر میں تعاملات اور تعلقات کا مرحلہ وار تجزیہ کریں۔

                        پیروی کرنے کے لیے مراحل:
                        1. تصویر میں تمام اکائیوں (اشیاء/لوگوں) کی شناخت کریں
                        2. اکائیوں کے درمیان مکانی تعلقات کا معائنہ کریں
                        3. کسی بھی براہ راست تعامل کا مشاہدہ کریں
                        4. فنکشنل تعلقات پر غور کریں
                        5. کسی بھی مضمر یا سماجی روابط کا تجزیہ کریں

                        ان مراحل کی پیروی کے بعد، فراہم کریں:
                        1. تمام تعاملات اور تعلقات کا آپ کا تفصیلی تجزیہ
                        2. آپ کی دلیل: وضاحت کریں کہ آپ نے ان تعلقات کی شناخت کیسے کی اور کون سے شواہد آپ کے نتائج کی تائید کرتے ہیں

                        مرحلہ وار سوچیں
                        """,
                                    4: """آئیے اس تصویر کو مرحلہ وار معنی خیز علاقوں میں تقسیم کریں۔

                        پیروی کرنے کے لیے مراحل:
                        1. تصویر میں بڑے مکانی تقسیمات کی شناخت کریں
                        2. ہر علاقے کی فنکشن یا فطرت کا تعین کریں
                        3. ہر علاقے کے اندر موجود مواد کو نوٹ کریں
                        4. مشاہدہ کریں کہ علاقے کیسے جڑتے ہیں یا منتقل ہوتے ہیں
                        5. ہر علاقے کی درجہ بندی اہمیت پر غور کریں

                        ان مراحل کی پیروی کے بعد، فراہم کریں:
                        1. تصویر کے معنی خیز علاقوں کا آپ کا تفصیلی تجزیہ 
                        2. آپ کی دلیل: وضاحت کریں کہ آپ نے ان علاقوں کا تعین کیسے کیا اور انہیں اس طرح کیوں منظم کیا

                        مرحلہ وار سوچیں
                        """,
                                    5: """آئیے اس تصویر کی ایک تفصیلی بیانیہ تفصیل مرحلہ وار بنائیں۔

                        پیروی کرنے کے لیے مراحل:
                        1. مرکزی موضوع یا مرکز کو قائم کریں
                        2. فوری تناظر اور ماحول کی تفصیل بیان کریں
                        3. کسی بھی کارروائی یا حرکات کو نوٹ کریں
                        4. ظاہری شکل اور حالت کے بارے میں متعلقہ تفصیلات شامل کریں
                        5. وقت کے پہلوؤں پر غور کریں (اس لمحے تک کیا ہوا)

                        ان مراحل کی پیروی کے بعد، فراہم کریں:
                        1. آپ کی تفصیلی بیانیہ تفصیل
                        2. آپ کی دلیل: وضاحت کریں کہ آپ نے یہ بیانیہ کیسے تشکیل دیا اور کون سے اہم عناصر نے آپ کی تفصیل کو متاثر کیا

                        مرحلہ وار سوچیں
                        """,
                                    
                                    # Image-text tasks (6-8) with Chain of Thought
                                    6: """آئیے تصویر اور متن کے درمیان تعلق کا مرحلہ وار تجزیہ کریں تاکہ ملتے جلتے عناصر کی شناخت کی جا سکے۔

                        پیروی کرنے کے لیے مراحل:
                        1. سب سے پہلے، تصویر میں تمام بصری عناصر کا احتیاط سے مشاہدہ کریں
                        2. متن کو اچھی طرح پڑھیں تاکہ سمجھ سکیں کہ یہ کیا بیان کر رہا ہے
                        3. متن میں ذکر کردہ اہم اداروں/تصورات کی ذہنی فہرست بنائیں
                        4. متن میں ہر ادارے کے لیے، تصویر میں اس کی بصری نمائندگی تلاش کریں
                        5. نوٹ کریں کہ کون سے متنی عناصر کے واضح بصری تعلقات ہیں
                        6. صفات، اعمال، اور مکانی تعلقات جیسی تفصیلات پر توجہ دیں

                        ان مراحل کی پیروی کے بعد، فراہم کریں:
                        1. متن کے کون سے مخصوص حصے تصویر کے عناصر سے مماثل ہیں، اس کا آپ کا تفصیلی تجزیہ
                        2. آپ کی دلیل: واضح کریں کہ شناخت شدہ ہر متنی حصہ بصری عناصر سے کیسے مطابقت رکھتا ہے، مخصوص بصری شواہد کا حوالہ دیتے ہوئے

                        مرحلہ وار سوچیں
                        """,
                                    7: """آئیے مرحلہ وار شناخت کریں کہ متن کے کون سے حصے تصویر سے متعلق نہیں ہیں۔

                        پیروی کرنے کے لیے مراحل:
                        1. سب سے پہلے، تصویر میں موجود تمام بصری عناصر کی تفصیلی دستاویز بنائیں
                        2. متن کو احتیاط سے پڑھیں اور تمام مختلف بیانات یا دعووں کی شناخت کریں
                        3. ہر متنی عنصر کے لیے، متعلقہ بصری شواہد کی فعال طور پر تلاش کریں
                        4. دو فہرستیں بنائیں: بصری تعاون والے متنی عناصر اور بصری تعاون کے بغیر متنی عناصر
                        5. بغیر بصری تعاون والے عناصر کے لیے، اس بات کی تصدیق کرنے کے لیے تصویر کی دوبارہ جانچ کریں کہ وہ واقعی غیر موجود ہیں
                        6. غور کریں کہ آیا کوئی متنی عناصر مضمر ہو سکتے ہیں لیکن براہ راست نظر نہیں آتے

                        ان مراحل کی پیروی کے بعد، فراہم کریں:
                        1. متن کے کون سے مخصوص حصے تصویر سے متعلق نہیں ہیں، اس کا آپ کا تفصیلی تجزیہ 
                        2. آپ کی دلیل: وضاحت کریں کہ یہ عناصر غیر متعلقہ کیوں ہیں، اس بات کی وضاحت کرکے کہ ان متنی عناصر کی حمایت کے لیے تصویر میں کیا موجود ہونا چاہیے

                        مرحلہ وار سوچیں
                        """,
                                    8: """آئیے تصویر اور متن میں ذکر کردہ مقامات کی مرحلہ وار شناخت اور تجزیہ کریں۔

                        پیروی کرنے کے لیے مراحل:
                        1. سب سے پہلے، تصویر میں نظر آنے والے کسی بھی مقامات/جگہوں کی شناخت کریں
                        2. متن میں کسی بھی جگہوں یا مقامات کے واضح ذکر کی تلاش کریں
                        3. شناخت کردہ ہر جگہ کے لیے، تعین کریں کہ یہ تصویر، متن، یا دونوں میں ظاہر ہوتی ہے
                        4. ایسی جگہوں کے لیے جو مشہور ہو سکتی ہیں، تشخیص کریں کہ آیا ان کی ثقافتی، تاریخی، یا عام اہمیت ہے
                        5. ہر جگہ کی شناخت کی حمایت کرنے والے شواہد اکٹھا کریں
                        6. تناظر پر غور کریں تاکہ یہ تعین کیا جا سکے کہ یہ جگہیں مواد کے لیے کیوں متعلقہ ہیں

                        ان مراحل کی پیروی کے بعد، فراہم کریں:
                        1. تصویر یا متن میں ذکر کردہ تمام جگہوں کا آپ کا تفصیلی تجزیہ
                        2. ہر جگہ کے لیے، واضح کریں کہ یہ تصویر، متن، یا دونوں میں ظاہر ہوتی ہے
                        3. مشہور جگہوں کے لیے، ان کی اہمیت کی وضاحت کریں
                        4. آپ کی دلیل: وضاحت کریں کہ آپ نے ہر جگہ کی شناخت کیسے کی اور کون سے شواہد آپ کے نتائج کی حمایت کرتے ہیں

                        مرحلہ وار سوچیں
                        """
                                }
    }
}


# =======================================================================
# HELPER FUNCTIONS (Same as before)
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

def parse_response_and_rationale(response_text):
    """Splits the model output into the main response and the rationale."""
    # This assumes the model consistently uses a phrase like "Your rationale:"
    # We use case-insensitive splitting for robustness.
    # The split is limited to 1 to handle cases where "rationale" might appear in the response itself.
    parts = response_text.lower().split("your rationale:", 1)
    if len(parts) > 1:
        # Find the original index to preserve casing in the main response
        split_index = response_text.lower().find("your rationale:")
        analysis = response_text[:split_index].strip()
        rationale = response_text[split_index + len("your rationale:"):].strip()
    else:
        analysis = response_text.strip()
        rationale = "No explicit rationale provided."
    return analysis, rationale

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
    """Main function to run the inference process with rationales."""
    
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
    print(f"Starting VLM Inference: Zero-Shot with Rationales")
    print("=====================================================")
    print(f"  Model:      {args.model}")
    print(f"  Language:   {args.language}")
    print(f"  Task:       {args.task}")

    # --- Prepare File Paths and Checkpoints ---
    data_path = os.path.join(args.data_dir, lang_config['code'])
    output_filename = os.path.join(args.output_dir, f"{args.model}_{args.language}_task{args.task}_zeroshot_with_rationales.json")
    checkpoint_filename = os.path.join(args.checkpoint_dir, f"ckpt_{args.model}_{args.language}_task{args.task}_zeroshot_with_rationales.json")
    
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
        prompt_template = lang_config["prompt_template_image_text"]
        print(f"\nProcessing {len(files_to_process)} image-text pairs...")

        for filename in tqdm(files_to_process, desc="Processing Image-Text Pairs"):
            image_id = get_image_id(filename)
            image_path = os.path.join(data_path, filename)
            text_path = os.path.join(data_path, f"text{image_id}.txt")
            
            if not os.path.exists(text_path):
                continue

            base64_image = encode_image(image_path)
            text_content = read_text_file(text_path)
            if not base64_image or text_content is None:
                continue

            formatted_prompt = prompt_template.format(
                task_description=task_description,
                text_content=text_content
            )

            messages = [{"role": "system", "content": lang_config["system_prompt"]},
                        {"role": "user", "content": [{"type": "text", "text": formatted_prompt},
                                                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
            
            result = process_batch(client, args.model, messages, args.max_tokens, args.temperature)
            if result:
                analysis, rationale = parse_response_and_rationale(result)
                results[image_id] = {"id": image_id, f"Task_{args.task}": analysis, f"Rationale_{args.task}": rationale}

            with open(checkpoint_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    else: # Image-only tasks
        prompt = lang_config["prompt_template_image_only"].format(task_description=task_description)
        print(f"\nProcessing {len(files_to_process)} images in batches of {BATCH_SIZE}...")
        
        for i in tqdm(range(0, len(files_to_process), BATCH_SIZE), desc="Processing Image Batches"):
            batch_filenames = files_to_process[i:i+BATCH_SIZE]
            
            content = [{"type": "text", "text": prompt}]
            valid_filenames = []
            for filename in batch_filenames:
                base64_image = encode_image(os.path.join(data_path, filename))
                if base64_image:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                    valid_filenames.append(filename)

            if len(content) == 1: continue

            messages = [{"role": "system", "content": lang_config["system_prompt"]},
                        {"role": "user", "content": content}]
            
            result = process_batch(client, args.model, messages, args.max_tokens, args.temperature)
            if result:
                analysis, rationale = parse_response_and_rationale(result)
                for filename in valid_filenames:
                    image_id = get_image_id(filename)
                    results[image_id] = {"id": image_id, f"Task_{args.task}": analysis, f"Rationale_{args.task}": rationale}

            with open(checkpoint_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # --- Final Save ---
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\nProcessing complete. Final results saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM inference with rationales for the VLURes benchmark.")
    
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

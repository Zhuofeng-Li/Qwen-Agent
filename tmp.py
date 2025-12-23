import json
import os
import asyncio
from dotenv import load_dotenv
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents import AsyncAssistant
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

load_dotenv()

# Prompt templates
SYSTEM_INSTRUCTION = "You are a document search assistant. Find the MINIMAL set of documents to answer the question. Output final response in JSON format."


USER_PROMPT_TEMPLATE = """Question: {query}
Expected Answer: {answer}

Available Search Results:
{organic_info}

TASK:
Scrape documents sequentially (1→2→3...) until you have enough information to answer the question and the answer is exactly the same as the expected answer.

PROCESS:
1. Scrape position 1
2. Evaluate: "Can I answer the question with current documents?"
   → YES: STOP IMMEDIATELY and go to OUTPUT step (DO NOT scrape more documents)
   → NO: Scrape position 2
3. Repeat step 2 for positions 3, 4, 5...
4. If all documents exhausted without sufficient info: Go to OUTPUT step

CRITICAL STOPPING RULE:
- IMMEDIATELY STOP scraping once you have sufficient information to answer the question
- DO NOT continue scraping additional documents after you can answer the question
- The goal is to find the MINIMUM number of documents needed
- Scraping unnecessary documents wastes resources and is incorrect behavior

CRITICAL - YOU MUST END WITH THIS OUTPUT:
After you finish scraping and evaluation, you MUST output the following JSON format as your final response:

{{
    "query": "{query}",
    "answer": "{answer}",
    "is_sufficient": true or false,
    "reasoning": "Why the scraped documents are sufficient/insufficient to answer the question"
}}

RULES:
- Never skip positions (must go 1→2→3→4 in order)
- STOP IMMEDIATELY when you have enough information - minimize document count
- If is_sufficient = true, the reasoning should explain why the current set of documents is sufficient, and specify which documents and what information within them can be used to answer the question
- If is_sufficient = false (only if all documents are insufficient), the reasoning should explain why the current set of documents is insufficient, and specify what information is missing

MANDATORY: Your final message MUST be ONLY the JSON object above. No additional text before or after the JSON.
"""

# Step 1: Load serper API results from JSONL file
def load_serper_results(file_path):
    """
    Load serper search results from a JSONL file.

    Args:
        file_path: Path to the JSONL file containing serper search results

    Returns:
        List of dictionaries, each containing query, answer, and search_results
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

# Load the data
serper_data_path = '/home/ubuntu/jianwen-us-midwest-1/panlu/zhuofeng-tamu/tevatron/gpt_oss_data_process/data/serper_search_results.jsonl'
serper_results = load_serper_results(serper_data_path)[:80] # TODO: remove this limit

# Step 2: Define serper_scrape tool for qwen_agent
@register_tool('serper_scrape')
class SerperScrapeTool(BaseTool):
    description = 'Scrape webpage content using Serper API. Input: URL of the webpage. Output: JSON string with webpage content including text.'
    parameters = {
        'type': 'object',
        'properties': {
            'url': {
                'type': 'string',
                'description': 'The URL of the webpage to scrape',
            }
        },
        'required': ['url'],
    }

    async def call(self, params: str, **kwargs) -> str:
        """
        Call the tool with the given parameters (async version).

        Args:
            params: JSON string containing the parameters

        Returns:
            The webpage content as a JSON string
        """
        import json5
        import aiohttp

        params_dict = json5.loads(params)
        url = params_dict['url']

        api_url = "https://scrape.serper.dev/"

        payload = {
            "url": url
        }

        headers = {
            'X-API-KEY': os.getenv('SERPER_API_KEY'),
            'Content-Type': 'application/json'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30.0)) as response:
                    response.raise_for_status()
                    result_json = await response.json()

            # Extract key information
            simplified = {
                'url': url,
                'text': result_json.get('text', ''),
            }

            if simplified['text'] == '':
                print(f"[DEBUG] Error: {url} returned no text")

            return json.dumps(simplified, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR SerperScrape] {url}: {type(e).__name__}: {str(e)}")
            return json.dumps({
                'error': str(e),
                'url': url,
                'text': ''
            }, ensure_ascii=False)


# Step 3: Create agent and process search results
async def find_golden_docs(serper_item):
    """
    Find the minimal set of documents that answer the question.

    Args:
        serper_item: A dictionary containing query, answer, and search_results

    Returns:
        Dictionary with golden_docs and search summary
    """
    index = serper_item['index']
    query = serper_item['query']
    answer = serper_item['answer']
    organic_results = serper_item['search_results']['organic']

    # Create prompt with organic results (handle missing snippets)
    organic_info = "\n".join([
        f"{i+1}. [{item.get('title', 'No title')}]({item.get('link', 'No link')})\n   Snippet: {item.get('snippet', 'No snippet available')}"
        for i, item in enumerate(organic_results)
    ])

    user_prompt = USER_PROMPT_TEMPLATE.format(
        query=query,
        answer=answer,
        organic_info=organic_info
    )

    # Configure LLM for AsyncAssistant
    llm_cfg = {
        'model': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
        'model_type': 'oai_async',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
    }

    # Create AsyncAssistant with serper_scrape tool
    # Set files=[] to prevent Memory initialization which triggers dashscope warnings
    bot = AsyncAssistant(
        llm=llm_cfg,
        system_message=SYSTEM_INSTRUCTION,
        function_list=['serper_scrape'],
        files=[]  # Explicitly disable files to avoid Memory/dashscope initialization
    )

    # Run the agent
    messages = [{'role': 'user', 'content': user_prompt}]
    response = await bot.run(messages=messages)

    # Extract tool results and final response
    tool_results = []
    final_response = ""

    for msg in response:
        # Extract tool calls and results
        if msg.get('role') == 'function':
            import json5
            content = json5.loads(msg.get('content', '{}'))
            if 'url' in content:
                tool_results.append({
                    'url': content.get('url', ''),
                    'text': content.get('text', ''),
                })
        # Extract final assistant response
        elif msg.get('role') == 'assistant' and msg.get('content'):
            final_response = msg['content']

    return {
        'index': index,
        'query': query,
        'answer': answer,
        'search_results': organic_results,
        'scraped_results': tool_results,
        'final_response': final_response
    }


# Step 4: Main async function
async def main(max_workers=32):
    """
    Main async function to process serper search results.

    Args:
        max_workers: Number of concurrent tasks
    """
    print(f"Loaded {len(serper_results)} serper search results\n")

    output_file = '/home/ubuntu/jianwen-us-midwest-1/panlu/zhuofeng-tamu/tevatron/gpt_oss_data_process/data/serper_browse_results.jsonl'
    error_file = '/home/ubuntu/jianwen-us-midwest-1/panlu/zhuofeng-tamu/tevatron/gpt_oss_data_process/data/serper_browse_errors.jsonl'

    # Check for already processed items
    processed_indices = set()
    if os.path.exists(output_file): 
        print(f"Found existing results file")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    processed_indices.add(result['index'])
        print(f"Already processed: {len(processed_indices)}")

    # Filter items to process
    items_to_process = [item for item in serper_results if item['index'] not in processed_indices]

    print(f"Total: {len(serper_results)} | To process: {len(items_to_process)}\n")

    if not items_to_process:
        print("All items processed!")
        return []

    # Process with semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_workers)

    # Progress tracking
    success_count = 0
    error_count = 0
    pbar = tqdm(total=len(items_to_process), desc="Processing", unit="item")

    async def process_item(item):
        nonlocal success_count, error_count
        async with semaphore:
            try:
                result = await find_golden_docs(item)

                # Save immediately
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

                success_count += 1
                pbar.set_postfix({'Success': success_count, 'Error': error_count})
                pbar.update(1)
                return result
            except Exception as e:
                error_result = {
                    'index': item.get('index', 'unknown'),
                    'query': item.get('query', ''),
                    'error': str(e)
                }
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                error_count += 1
                pbar.set_postfix({'Success': success_count, 'Error': error_count})
                pbar.update(1)
                tqdm.write(f"[✗] Index {item.get('index')}: {str(e)}")
                return None

    # Run all tasks concurrently
    tasks = [process_item(item) for item in items_to_process]
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r]

    pbar.close()

    print(f"\n{'='*80}")
    print(f"Completed: {len(results)}/{len(items_to_process)}")
    print(f"Results: {output_file}")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    asyncio.run(main(max_workers=128))

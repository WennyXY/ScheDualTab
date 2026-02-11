from datetime import datetime
import json
import asyncio
import tqdm
import argparse
from model.self_consistency_tqa import TableQASingleHop

from llama_index.llms.ollama import Ollama

from data.wikiQA import WikiQA
from data.tabFact import TabFact
import utils.utils as utils

def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    # Add arguments here
    parser.add_argument('--data_path', type=str, default="WikiTableQuestions/", help='Path to the dataset')
    parser.add_argument('--table_path', type=str,default="data/pristine-unseen-tables.tsv", help='Path to the testset')
    parser.add_argument('--output_path', type=str,default="", help='Path to the answer file')
    parser.add_argument('--prompt_file', type=str,default="prompt/prompt.yaml", help='Path to the prompt')
    parser.add_argument('--llm', type=str,default='qwen3:4b', help='LLM model used')
    parser.add_argument('--s', type=int,default='0')
    parser.add_argument('--e', type=int,default='-1')
    parser.add_argument('--textual_path_num', type=int, default='3')
    parser.add_argument('--symbolic_path_num', type=int, default='2')
    parser.add_argument('--base_url', type=str, default='', help='LLM base url')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--aggregation', type=str, default="fuzzy_match")

    return parser.parse_args()

def load_llm(model_name, base_url):
    return Ollama(
        model = model_name,
        temperature = 0.7,
        request_timeout=3600,
        base_url=base_url,
        max_retries=10      
    )

def load_dataset(data_path, table_path):
    if 'WikiTableQuestions' in data_path:
        return WikiQA(data_path, table_path)
    elif 'tabfact' in data_path.lower():
        return TabFact(data_path, table_path)
    return None

def load_workflow(llm, prompt_file, aggregation='fuzzy_match', textual_reasoning_prompt = 'text_prompt_str'):
    prompt_dict = utils.get_prompt_dict(prompt_file)
    workflow = TableQASingleHop(llm=llm, prompt_dict=prompt_dict, timeout=3600, verbose=True, aggregation=aggregation, textual_reasoning_prompt = textual_reasoning_prompt)
    return workflow

def setup(args):
    llm = load_llm(args.llm, args.base_url)
    data = load_dataset(args.data_path, args.table_path)
    textual_reasoning_prompt = 'text_prompt_str'
    if 'tabfact' in args.data_path.lower():
        textual_reasoning_prompt = 'tabfact_prompt_oneshot'

    workflow = load_workflow(llm, args.prompt_file, args.aggregation, textual_reasoning_prompt)
    
    return data, workflow

async def run_tableQA(start, end, args):
    now = datetime.now()
    dt_string = now.strftime("%m-%d_%H:%M")
    tableqa, tableQAworkflow = setup(args)
    # Save the responses
    subset = args.table_path.split('.')[0].split('/')[1]
    if args.output_path != '':
        output_path = args.output_path.replace('.json',f'_s{start}_e{end}_{dt_string}.json')
    else:
        output_path = f'outputs/{subset}_s{start}_e{end}_{args.llm}_{dt_string}.json'
    print(f"Output file: {output_path}")
    fw = open(output_path, 'w')
    
    index = start
    print(f"total test set: {len(tableqa.data)}")
    for entry in tqdm.tqdm(tableqa.data[start:end]):
        id, table, intro, question, answer = tableqa.get_table_info(entry)
        entry['table_id'] = id
        try:
            response = await tableQAworkflow.run(query=question, table_id = entry['table_id'], table=table, textual_path_num=args.textual_path_num, symbolic_path_num=args.symbolic_path_num, retrieve_num=1, table_context = intro)
        except:
            response = await tableQAworkflow.run(query=question, table_id = entry['table_id'], table=table, textual_path_num=args.textual_path_num, symbolic_path_num=args.symbolic_path_num, retrieve_num=1, table_context = intro)

        response = response.split("|")
        response = ', '.join(response)
        print(f"gt: {answer} re:{response}")
        tmp = {'index': index, 'question_id': id, 'table_id': entry['table_id'],'question': question, 'response': response, 'answer': answer}
        index += 1
        fw.write(json.dumps(tmp) + '\n')
    fw.close()    
    print("\n============\n")


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    asyncio.run(run_tableQA(args.s,args.e,args))
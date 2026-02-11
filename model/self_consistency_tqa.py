from typing import Any, List, Optional, Dict
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
from llama_index.core.prompts import PromptTemplate

import pandas as pd
from llama_index.core.llms import LLM
from llama_index.core.output_parsers.base import ChainableOutputParser

import utils.utils as utils

class FinalAnswerOutputParser(ChainableOutputParser):
    def parse(self, output: str, split_content='Final Answer:') -> Optional[str]:
        if "</think>" in output:
            output = output.split("</think>")[-1]
        if split_content in output:
            lines = output.split(split_content)
            if len(lines) > 1:
                return lines[-1].strip()
        return output

    def format(self, query: Any) -> str:
        return query

class RetrivalOutputParser(ChainableOutputParser):
    def parse(self, output: str, split_content=None) -> Optional[str]:
        if "</think>" in output:
            output = output.split("</think>")[-1].strip()
        # print(output)
        if split_content and split_content in output:
            output = output.split(split_content)[-1].strip()
        return output

    def format(self, query: Any) -> str:
        return query
    

def output_preprocessor(output):
    if '</think>' in output:
        output = output.split('</think>')[1]
    output = output.replace('python', '')
    return output.strip()

class QueryEvent(Event):
    table_context: str

class TextualReasoningEvent(Event):
    table: str
    table_context: str
    table_schema: str
    query: str

class AugTextualReasoningEvent(Event):
    table: pd.DataFrame
    table_context: str
    table_schema: str
    query: str

class ResultEvent(Event):
    result: str | int

class TableQASingleHop(Workflow):
    def __init__(self, llm: LLM|None=None, prompt_dict: Dict ={},aggregation='voting', textual_reasoning_prompt = 'text_prompt_str', **workflow_kwargs: Any,)-> None:
        super().__init__(**workflow_kwargs)
        self.llm = llm
        self.prompt_dict = prompt_dict
        self.textual_reasoning_prompt = textual_reasoning_prompt
        print('textual_reasoning_prompt: ', prompt_dict[textual_reasoning_prompt])
        if 'fuzzy' in aggregation:
            print('fuzzy match')
            self.aggregation = utils.aggregate_sc_by_fuzzy_match
        else:
            print('majority voting')
            self.aggregation = utils.aggregate_sc_by_voting

    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        # load data
        await ctx.set("query", ev.get("query"))
        await ctx.set("table", ev.get("table"))
        await ctx.set("textual_path_num", ev.get("textual_path_num"))
        await ctx.set("symbolic_path_num", ev.get("symbolic_path_num"))
        table_context = ev.table_context
        
        if table_context != '' and table_context != None:
            table_context = await self.context_summarisation(context=table_context)
        print(f"table_context: {table_context}")
        return QueryEvent(table_context=table_context)

    def _get_table_schema(self, table_df):
        df_schema = "column name (type)\n"
        for col in table_df.columns:
            dtype = table_df[col].dtype
            example = table_df[col].iloc[0]
            df_schema += f"{col} ({dtype} example: {example})\n"
        return df_schema
    
    async def generate_table_schema(self, table_df, table_context):
        prompt = PromptTemplate(self.prompt_dict['get_table_schema'])
        retrieved_context = await self.llm.apredict(
            prompt = prompt,
            table_md = table_df.to_markdown(),
            table_info = table_context
        )
        parser = RetrivalOutputParser()
        table_schema = parser.parse(retrieved_context)
        print(f"table_schema: {table_schema}")
        return table_schema
    
    async def context_summarisation(self, context):
        prompt = PromptTemplate(self.prompt_dict['info_summarisation_prompt_str'])
        retrieved_context = await self.llm.apredict(
            prompt = prompt,
            table_info = context
        )
        parser = RetrivalOutputParser()
        return parser.parse(retrieved_context)

    async def query_rewrite(self, query, table_context, table_schema, table_df=None):
        # paraphrase query based on table info and original query
        # query_rewrite_prompt
        # logic_query_rewrite
        # schema_align_query_rewrite
        # query_format_prompt
        # sql_query_rewrite
        # nl_query_rewrite
        # hints_augment
        # query_enhancement
        prompt = PromptTemplate(self.prompt_dict['query_enhancement_new'])
        response = await self.llm.apredict(
            prompt = prompt,
            table_info = table_context,
            table_schema = table_schema,
            query = query,
            table_exp = "Table Preview: \n"+table_df.head(2).to_markdown()
        )
        output_parser = RetrivalOutputParser()
        response = output_parser.parse(response, 'Rewritten Query:')
        print('query_rewrite:', response)
        return response
    
    @step
    async def process_query(self, ctx: Context, ev: QueryEvent) -> TextualReasoningEvent | AugTextualReasoningEvent | None:
        table_context = ev.table_context
        query = await ctx.get("query")
        table = await ctx.get("table")
        textual_path_num = await ctx.get("textual_path_num")
        rewritten_query_num = await ctx.get("symbolic_path_num")
        
        table_schema = await self.generate_table_schema(table, table_context)

        # ======= base query =======
        for i in range(textual_path_num):
            if i >= textual_path_num/2:
                print('base table.')
                ctx.send_event(TextualReasoningEvent(table_context=table_context, table=table.to_markdown(), table_schema=table_schema, query=query))
            else:
                # print('sentence table. base schema')
                print('base table. base schema')
                ctx.send_event(TextualReasoningEvent(table_context=table_context, table=sentence_table.to_markdown(), table_schema=table_schema, query=query))
        
        # ======= augmented query =======
        for i in range(rewritten_query_num):
            print('base table. base schema(aug)')
            ctx.send_event(AugTextualReasoningEvent(table_context=table_context, table=table, table_schema=table_schema, query=query))

        return None
       
    @step(num_workers=6)
    async def textual_reasoning(self, ev: TextualReasoningEvent) -> ResultEvent:
        print('> textual reasoning...')
        table_context = ev.table_context
        table_schema = ev.table_schema
        query = ev.query
        table = ev.table
        
        if table_context == "" or table_context == None:
            table_context = table_schema
        prompt = PromptTemplate(self.prompt_dict[self.textual_reasoning_prompt])
        response = await self.llm.apredict(
            prompt = prompt,
            table_md = table,
            table_info = table_context,
            query = query
        )
        output_parser = FinalAnswerOutputParser()
        response = output_parser.parse(response)
        if self._verbose:
            print(f"> Textual Output: {query}: {response}\n")
        if not response:
            response = ''
        return ResultEvent(result=response)
    
    @step(num_workers=5)
    async def augmented_textual_reasoning(self, ev: AugTextualReasoningEvent) -> ResultEvent:
        print('> Aug textual reasoning...')
        table_context = ev.table_context
        table_schema = ev.table_schema
        query = ev.query
        table = ev.table
        
        enhancement = await self.query_rewrite(query, table_context, table_schema, table)
        processed_query = f"{query} \nReasoning Hints (extracted automatically, may contain errors.): \n{enhancement}\nAlways verify hints against the table before following. \n"

        if table_context == "" or table_context == None:
            table_context = table_schema
        prompt = PromptTemplate(self.prompt_dict[self.textual_reasoning_prompt])
        response = await self.llm.apredict(
            prompt = prompt,
            table_md = table.to_markdown(),
            table_info = table_context,
            query = processed_query
        )
        output_parser = FinalAnswerOutputParser()
        response = output_parser.parse(response)
        if self._verbose:
            print(f"> Textual Output(Aug): {processed_query}: {response}\n")
        if not response:
            response = ''
        return ResultEvent(result=response)
    
    @step
    async def combine_results(
        self, ctx: Context, ev: ResultEvent
    ) -> StopEvent | None:
        textual_path_num = await ctx.get("textual_path_num")
        symbolic_path_num = await ctx.get("symbolic_path_num")
        
        results = ctx.collect_events(ev, [ResultEvent] * (textual_path_num +symbolic_path_num))
        
        if results is None:
            print('Results None, Processing...')
            return None
        print(f"Results Num: {len(results)}")
        combined_result = self.aggregation([event.result for event in results])
        return StopEvent(result=combined_result)
    
    def _aggregate_self_consistency(self, results: List[str]) -> str:
        counts = {}
        for result in results:
            if result in counts:
                counts[result] += 1
            else:
                counts[result] = 1
        return max(counts, key=counts.get)
    
    def _get_df_str(self, table_df):
        return str(table_df.head(len(table_df)))
    
    def get_sentence_table(self, table_df):
        print("> Serialising....")
        sentence_table = table_df.apply(utils.row_to_sentence, args=(None, None), axis=1)
        sentence_table = sentence_table.to_frame()
        
        return sentence_table

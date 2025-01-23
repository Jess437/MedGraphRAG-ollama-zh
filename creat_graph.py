import os
import argparse
from getpass import getpass

from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO

# from langchain.output_parsers.openai_tools import JsonOutputToolsParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda

from langchain_community.chat_models import ChatOpenAI
# from langchain.chains import create_extraction_chain
# from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_helper.ollama_functions import OllamaFunctions
from langchain_helper.extraction import create_extraction_chain

from langchain_core.pydantic_v1 import BaseModel
from langchain import hub

import os
from typing import Optional, List
from agentic_chunker import AgenticChunker
from config import config
from utils import *

# Pydantic data class
class Sentences(BaseModel):
    sentences: List[str]


def get_propositions(text, runnable, extraction_chain):
    runnable_output = runnable.invoke({
    	"input": text
    }).content
    
    # propositions = extraction_chain.run(runnable_output)[0].sentences
    
    print("###########")
    print("Runnable Output: ", runnable_output)
    print("###########")
    propositions = extraction_chain.run(runnable_output)[0]["sentences"]
    print("Propositions: ", propositions)
    print("############")    
    return propositions

def run_chunk(essay):

    obj = hub.pull("wfh/proposal-indexing")
    # llm = ChatOpenAI(model=config.model, openai_api_key = os.getenv("OPENAI_API_KEY"))
    llm = OllamaFunctions(
        model=config.model,
        base_url=config.base_url,
        temperature=0,
    )

    runnable = obj | llm

    # # Extraction
    # extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)
    
    extraction_chain = create_extraction_chain(
        {
            "properties": {
                "sentences": {
                    "title": "Sentences",
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["sentences"]
        }, llm=llm
    )

    paragraphs = essay.split("\n\n")

    essay_propositions = []

    # for i, para in enumerate(paragraphs):
    #     propositions = get_propositions(para, runnable, extraction_chain)
        
    #     essay_propositions.extend(propositions)
    #     # print (f"Done with {i}")

    max_retries = 5  # 最大重試次數

    for i, para in enumerate(paragraphs):
        retry_count = 0
        while retry_count < max_retries:
            try: 
                propositions = get_propositions(para, runnable, extraction_chain)
                break
            except Exception as e:
                # 捕獲異常並增加重試計數
                retry_count += 1
                print(f"Error processing paragraph {i}, attempt {retry_count}: {e}")
        
        essay_propositions.extend(propositions)

    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type='list_of_strings')

    return chunks
    print(chunks)

from camel.models.ollama_model import OllamaModel
def creat_metagraph(args, content, gid, n4j):

    # Set instance
    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent(
        model=OllamaModel(
            model_type=config.model,
            model_config_dict={},
            url=f"{config.base_url}/v1",
            # temperature=0,
        )
    )
    whole_chunk = content

    if args.grained_chunk == True:
        content = run_chunk(content)
    else:
        content = [content]
    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        ans_str = kg_agent.run(element_example, parse_graph_elements=False)
        # print(ans_str)

        graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    if args.ingraphmerge:
        merge_similar_nodes(n4j, gid)
    add_sum(n4j, whole_chunk, gid)
    return n4j


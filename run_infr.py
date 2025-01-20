import os
from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
import argparse
from summerize import process_chunks
from tqdm import tqdm
from utils import *
from config import config

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-simple', action='store_true')
parser.add_argument('-construct_graph', action='store_true')
parser.add_argument('-inference',  action='store_true')
parser.add_argument('-grained_chunk',  action='store_true')
parser.add_argument('-trinity', action='store_true')
parser.add_argument('-trinity_gid1', type=str)
parser.add_argument('-trinity_gid2', type=str)
parser.add_argument('-ingraphmerge',  action='store_true')
parser.add_argument('-crossgraphmerge', action='store_true')
# parser.add_argument('-dataset', type=str, default='mimic_ex')

parser.add_argument('-data', type=str, default='./dataset_test.csv')
parser.add_argument('-len', type=int, default=10)


parser.add_argument('-test_data_path', type=str, default='./dataset_ex/report_0.txt')
args = parser.parse_args()

from openai import OpenAI

def get_response(n4j, gid, query):
    selfcont = ret_context(n4j, gid)
    linkcont = link_context(n4j, gid)
    
    user_one = "使用者的問題為: " + query + "\n" + \
               "以下是提供參考的資訊: " +  str(selfcont)
    res = call_llm(sys_prompt_one,user_one)
    
    user_two = "使用者的問題為: " + query + "\n" + \
               "你上次的回答是: " +  res + "\n" + \
               "提供的參考: " +  " ".join(linkcont)           
    res = call_llm(sys_prompt_two,user_two)
    
    print("#" * 100)
    
    print(selfcont)
    print(linkcont)
    
    print("#" * 100)
    
    print("user_one:")
    print(user_one)
    
    print("user_two:")
    print(user_two)
    
    print("llm returns:")
    print(res)
    
    print("#" * 100)
    
    return res


sys_p = """
Assess the similarity of the two provided summaries and return a rating from these options: 'very similar', 'similar', 'general', 'not similar', 'totally not similar'. Provide only the rating.
"""

# def seq_ret(n4j, sumq):
#     rating_list = []
#     sumk = []
#     gids = []
#     sum_query = """
#         MATCH (s:Summary)
#         RETURN s.content, s.gid
#         """
#     res = n4j.query(sum_query)
#     for r in res:
#         sumk.append(r['s.content'])
#         gids.append(r['s.gid'])
    
#     for sk in sumk:
#         sk = sk[0]
#         rate = call_llm(sys_p, "The two summaries for comparison are: \n Summary 1: " + sk + "\n Summary 2: " + sumq[0])
#         rate = rate.lower()
        
#         print("llm returns", rate, "for summary", sk)
        
#         if "totally not similar" in rate:
#             rating_list.append(0)
#         elif "not similar" in rate:
#             rating_list.append(1)
#         elif "general" in rate:
#             rating_list.append(2)
#         elif "very similar" in rate:
#             rating_list.append(4)
#         elif "similar" in rate:
#             rating_list.append(3)
#         else:
#             print("llm returns no relevant rate")
#             rating_list.append(-1)

#     ind = find_index_of_largest(rating_list)
#     # print('ind is', ind)

#     gid = gids[ind]

#     return gid

def seq_ret(n4j, sumq):
    rating_list = []
    sumk = []
    gids = []
    sum_query = """
        MATCH (s:Summary)
        RETURN s.content, s.gid
        """
    res = n4j.query(sum_query)
    for r in res:
        sumk.append(r['s.content'])
        gids.append(r['s.gid'])
    
    print("Performing llm on summaries")
    for sk in tqdm(sumk):
        sk = sk[0]
        rate = call_llm(sys_p, "The two summaries for comparison are: \n Summary 1: " + sk + "\n Summary 2: " + sumq[0])
        rate = rate.lower()
        
        if "totally not similar" in rate:
            rating_list.append(0)
        elif "not similar" in rate:
            rating_list.append(1)
        elif "general" in rate:
            rating_list.append(2)
        elif "very similar" in rate:
            rating_list.append(4)
        elif "similar" in rate:
            rating_list.append(3)
        else:
            print("llm returns no relevant rate")
            rating_list.append(-1)

        if rating_list[-1] > 2: 
            print("llm returns")
            print(rate)
            print("for summary")
            print(sk)
            
        
    ind = find_index_of_largest(rating_list)
    gid = gids[ind]

    return gid


def load_content(datapath):
    all_content = ""  # Initialize an empty string to hold all the content
    with open(datapath, 'r', encoding='utf-8') as file:
        for line in file:
            all_content += line.strip() + "\n"  # Append each line to the string, add newline character if needed
    return all_content


if __name__ == '__main__':
    print("###########")
    print("Start running")
    print("Arguments:")
    print(args)
    print("###########")
    
    # Set Neo4j instance
    n4j = Neo4jGraph(
        url=config.neo4j_url,
        username=config.neo4j_user,
        password=config.neo4j_pass
    )

    question = load_content("./prompt.txt")
    sum = process_chunks(question)
    gid = seq_ret(n4j, sum)
    response = get_response(n4j, gid, question)
    print(response)
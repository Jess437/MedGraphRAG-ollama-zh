import os
from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
import argparse
from creat_graph import creat_metagraph
from utils import *
from config import config
from dataloader import content_generator

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
parser.add_argument('-len', type=int, default=1000)
parser.add_argument('-start_idx', type=int, default=0)


args = parser.parse_args()

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
    
    if args.construct_graph: 
        # with open(args.data, newline='') as csvfile:
        #     reader = csv.DictReader(csvfile)
        #     # headers = next(reader)  # Skip the header row
            
        #     skip_rows = args.start_idx
        #     for i in range(skip_rows):
        #         next(reader)
                
        #     rows_processed = 0
            
        #     # Read and print the contents of each file
        #     for idx, row in enumerate(reader):
        #         if rows_processed >= args.len:
        #             break
        #         rows_processed += 1
                
        #         print(f"Processing row {idx+1}")
        #         print(row['summary'])
        #         content = "以下是屬於" + row['department'] + "的病例\n" + \
        #                   row['summary']
                
        #         print(content[:40])
        #         gid = str_uuid()
        #         n4j = creat_metagraph(args, content, gid, n4j)

        #         if args.trinity:
        #             link_context(n4j, args.trinity_gid1)
        #     if args.crossgraphmerge:
        #         merge_similar_nodes(n4j, None)

        if args.crossgraphmerge:
            merge_similar_nodes(n4j, None)
        
        for idx, content in enumerate(content_generator(args.data, start_idx=args.start_idx, length=args.len)):
            print("#" * 100)
            print(f"Running document {idx+1}")
            print(content[:40])
            gid = str_uuid()
            n4j = creat_metagraph(args, content, gid, n4j)

            if args.trinity:
                link_context(n4j, args.trinity_gid1)
                
        if args.crossgraphmerge:
            merge_similar_nodes(n4j, None)

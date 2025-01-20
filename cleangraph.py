# from neo4j import GraphDatabase

# class Neo4jConnection:
#     def __init__(self, uri, user, pwd):
#         self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

#     def close(self):
#         self.driver.close()

#     def clean_graph(self):
#         with self.driver.session() as session:
#             session.write_transaction(self._delete_all)

#     @staticmethod
#     def _delete_all(tx):
#         tx.run("MATCH (n) DETACH DELETE n")

# # Example usage
# conn = Neo4jConnection("bolt://host.docker.internal:7687", "neo4j", "binglabi")
# conn.clean_graph()
# conn.close()


from neo4j import GraphDatabase

def clean_graph(uri, user, pwd):
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as session:
        session.write_transaction(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
    driver.close()

# Example usage
response = input("Do you want to clean the graph? (y/n): ")

if response.lower() == "y":
    clean_graph("bolt://host.docker.internal:7687", "neo4j", "binglabi")
else:
    print("Graph will not be cleaned.")
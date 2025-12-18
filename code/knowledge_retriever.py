import time
from collections import defaultdict
from neo4j import GraphDatabase
from quickumls import get_quickumls_client

class KnowledgeRetriever:
    def __init__(self, neo4j_uri="bolt://localhost:7687", neo4j_auth=("neo4j", "admin"), quickumls_path=None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        # Assuming get_quickumls_client handles the path internally or via env vars as seen in test_qumls.py
        # If quickumls_path is meant to be passed to get_quickumls_client, it should be added here.
        # Based on test_qumls.py, it takes no args.
        self.matcher = get_quickumls_client() 
        self.ext_time_metrics = defaultdict(float)
        self.ext_time_metrics_counts = defaultdict(int)

    def extract_entities(self, text):
        seen = set()
        entities = {}
        # QuickUMLS match expects a list or single string? 
        # test_qumls.py uses: matches = matcher.match(text, ...) where text is a single string from a list loop.
        
        matches = self.matcher.match(text, best_match=True, ignore_syntax=False)
        for group in matches:
            if not group:
                continue
            top = group[0]
            ngram = top['ngram']
            if ngram not in seen:
                seen.add(ngram)
                entities[ngram] = top['cui']
        return entities

    def get_one_hop_neighbors(self, cuis):
        if not cuis:
            return {}

        _st_neo4j = time.time()
        # Query without relation filtering as requested
        query = """
        MATCH (c:Concept)-[r]->(neighbor:Concept)
        WHERE c.CUI IN $cui_list
        RETURN DISTINCT c.CUI AS source_cui,
            type(r) AS relation_type,
            r.RELA AS rela,
            neighbor.CUI AS neighbor_cui,
            neighbor.name AS neighbor_name
        """

        with self.driver.session() as session:
            results = session.run(query, cui_list=list(cuis))
            neighbors = defaultdict(set)
            for record in results:
                neighbors[record["source_cui"]].add((
                    record["rela"],
                    record["neighbor_name"],
                    record["neighbor_cui"]
                ))
        
        self.ext_time_metrics['cui_neo4j_get_neighbors'] += (time.time() - _st_neo4j)
        self.ext_time_metrics_counts['cui_neo4j_get_neighbors'] += 1
        return neighbors

    def close(self):
        self.driver.close()

"""
Analyze UMLS relation types in Neo4j
to understand what structural patterns are available
"""
from neo4j import GraphDatabase
from collections import Counter
import json

def analyze_relations():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "admin"))
    
    with driver.session() as session:
        # Get all distinct relation types
        print("=== Distinct Relation Types ===")
        result = session.run("""
            MATCH ()-[r]->()
            RETURN DISTINCT type(r) as rel_type, r.RELA as rela
            LIMIT 200
        """)
        
        rel_types = Counter()
        rela_types = Counter()
        
        for record in result:
            rel_types[record["rel_type"]] += 1
            if record["rela"]:
                rela_types[record["rela"]] += 1
        
        print("\nRelation Types (edge labels):")
        for rel, count in rel_types.most_common(20):
            print(f"  {rel}: {count}")
        
        print("\nRELA Types (semantic relations):")
        for rela, count in rela_types.most_common(50):
            print(f"  {rela}")
        
        # Get sample for each RELA type
        print("\n=== Sample Relations ===")
        result = session.run("""
            MATCH (a:Concept)-[r]->(b:Concept)
            WHERE r.RELA IS NOT NULL
            RETURN a.name as source, r.RELA as relation, b.name as target
            LIMIT 50
        """)
        
        print("\nSamples:")
        for record in result:
            print(f"  {record['source'][:30]:30} --[{record['relation']:20}]--> {record['target'][:30]}")
    
    driver.close()

if __name__ == "__main__":
    analyze_relations()

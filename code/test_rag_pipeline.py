from knowledge_retriever import KnowledgeRetriever
import sys

def test_retrieval():
    print("Initializing Knowledge Retriever...")
    try:
        retriever = KnowledgeRetriever()
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    text = "The patient presented with severe nausea and vomiting."
    print(f"\nTest Input: {text}")

    print("Extracting entities...")
    entities = retriever.extract_entities(text)
    print(f"Entities found: {entities}")

    if not entities:
        print("No entities found. Exiting.")
        retriever.close()
        return

    cuis = list(entities.values())
    print("\nRetrieving neighbors for CUIs:", cuis)
    
    neighbors = retriever.get_one_hop_neighbors(cuis)
    print(f"Neighbors found: {len(neighbors)} source concepts with neighbors.")
    
    for source_cui, rel_triples in neighbors.items():
        print(f"\nSource CUI: {source_cui}")
        # Show first 5 neighbors
        for i, (rela, n_name, n_cui) in enumerate(rel_triples):
            if i >= 5:
                print("... (more)")
                break
            print(f"  - [{rela}] -> {n_name} ({n_cui})")

    retriever.close()
    print("\nTest Complete.")

if __name__ == "__main__":
    test_retrieval()

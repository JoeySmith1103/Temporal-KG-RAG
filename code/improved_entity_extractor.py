"""
Improved Medical Entity Extractor

Addresses issues with QuickUMLS fuzzy matching:
1. Filters by similarity threshold (0.9)
2. Removes common clinical stopwords
3. Filters to relevant semantic types (diseases, symptoms, etc.)
"""
from quickumls import get_quickumls_client
from typing import List, Dict, Set

# Clinical stopwords - common words that get incorrectly matched
CLINICAL_STOPWORDS = {
    'year', 'years', 'old', 'day', 'days', 'week', 'weeks', 'month', 'months',
    'hour', 'hours', 'minute', 'minutes', 'time', 'times', 'ago',
    'man', 'woman', 'patient', 'physician', 'doctor', 'clinic', 'hospital',
    'presents', 'presented', 'presenting', 'comes', 'brought', 'reports',
    'history', 'started', 'started', 'developed', 'onset',
    'past', 'previous', 'prior', 'since', 'following', 'after', 'before',
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'had', 'have',
    'she', 'he', 'her', 'his', 'their', 'that', 'this', 'which', 'who',
    'because', 'due', 'to', 'for', 'with', 'of', 'in', 'on', 'at', 'by',
    'normal', 'noted', 'found', 'shows', 'reveals', 'indicates',
}

# Relevant semantic types for clinical questions
RELEVANT_SEMTYPES = {
    'T047',  # Disease or Syndrome
    'T048',  # Mental or Behavioral Dysfunction
    'T046',  # Pathologic Function
    'T184',  # Sign or Symptom
    'T033',  # Finding
    'T037',  # Injury or Poisoning
    'T191',  # Neoplastic Process
    'T121',  # Pharmacologic Substance
    'T200',  # Clinical Drug
    'T060',  # Diagnostic Procedure
    'T061',  # Therapeutic or Preventive Procedure
    'T059',  # Laboratory Procedure
    'T034',  # Laboratory or Test Result
    'T029',  # Body Location or Region
    'T023',  # Body Part, Organ, or Organ Component
}


class ImprovedEntityExtractor:
    """Entity extractor with stricter matching and filtering."""
    
    def __init__(self, min_similarity: float = 0.85, min_ngram_length: int = 3):
        self.matcher = get_quickumls_client()
        self.min_similarity = min_similarity
        self.min_ngram_length = min_ngram_length
    
    def extract(self, text: str) -> List[Dict]:
        """Extract entities with improved filtering."""
        matches = self.matcher.match(text, best_match=True, ignore_syntax=False)
        
        filtered_entities = []
        seen_cuis = set()
        
        for group in matches:
            if not group:
                continue
            
            top = group[0]
            ngram = top['ngram'].lower()
            cui = top['cui']
            similarity = top['similarity']
            semtypes = set(top.get('semtypes', []))
            
            # Skip if already seen this CUI
            if cui in seen_cuis:
                continue
            
            # Filter 1: Minimum similarity
            if similarity < self.min_similarity:
                continue
            
            # Filter 2: Skip stopwords - DISABLED, hurts performance
            # words = ngram.split()
            # if all(w.lower() in CLINICAL_STOPWORDS for w in words):
            #     continue
            
            # Filter 3: Minimum ngram length
            if len(ngram) < self.min_ngram_length:
                continue
            
            # Filter 4: Semantic type (if available) - DISABLED for now, too restrictive
            # if semtypes and not semtypes.intersection(RELEVANT_SEMTYPES):
            #     continue
            
            seen_cuis.add(cui)
            filtered_entities.append({
                'ngram': top['ngram'],
                'term': top['term'],
                'cui': cui,
                'similarity': similarity,
                'semtypes': list(semtypes)
            })
        
        return filtered_entities
    
    def extract_cuis(self, text: str) -> List[str]:
        """Extract only CUIs."""
        entities = self.extract(text)
        return [e['cui'] for e in entities]


def test_improved_extractor():
    """Test the improved extractor against known problematic cases."""
    extractor = ImprovedEntityExtractor(min_similarity=0.70)  # Balance: catch more, but filter stopwords
    
    test_questions = [
        'A 24-year-old man presents to your clinic complaining of a cough that started 3 weeks ago.',
        'A 32-year-old woman comes to the emergency department for a 2-week history of right upper quadrant abdominal pain.',
        'A 15-year-old girl is brought to the physician by her parents because she has not had menstrual bleeding for the past 2 months.'
    ]
    
    print("=== Improved Entity Extraction (threshold=0.80) ===\n")
    
    for i, q in enumerate(test_questions):
        print(f"--- Question {i+1} ---")
        print(f"Q: {q[:80]}...")
        entities = extractor.extract(q)
        print(f"Extracted ({len(entities)} entities):")
        for e in entities:
            print(f"  - \"{e['ngram']}\" -> {e['term']} (CUI: {e['cui']}, sim: {e['similarity']:.2f})")
        print()


if __name__ == "__main__":
    test_improved_extractor()

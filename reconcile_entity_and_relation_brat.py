import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os


from postprocessing.valid_combinations import valid_comb
schema = valid_comb

class PostProcessor:
    """
    A class to clean and enhance NER and RE annotations.
    It encapsulates the logic for merging fragmented entities and recovering
    missing relations based on a predefined schema.
    """

    def __init__(self, ner_annotations, re_annotations, raw_text, schema = schema, proximity_threshold=20, remove_backward_relations = True):
        """
        Initializes the PostProcessor with the raw data and configuration.

        Args:
            ner_annotations (list): The original list of NER entity tuples.
            re_annotations (list): The original list of RE relation tuples.
            schema (list): A list of valid [head_label, tail_label] combinations.
            proximity_threshold (int): Max character distance for generic entity merging.
        """
        # --- Store initial data ---
        self.raw_ner = ner_annotations
        self.raw_re = re_annotations
        self.raw_text = raw_text
        self.schema_set = {tuple(combo) for combo in schema}
        self.proximity_threshold = proximity_threshold
        self.remove_backward_relations = remove_backward_relations

        # --- Initialize variables to store results ---
        self.existing_re = []
        self.recovered_re = []

        self.merged_ner = []
        self.patched_ner = []

        self.final_re = []
        self.index_map = {} # Maps old NER indices to new ones after merging

    # --- Public Methods ---
    
    def process(self):
        """
        Executes the full post-processing pipeline in the correct order.
        This is the main entry point for using the class.
        """
        # print("Starting post-processing pipeline...")
        # 1. Merge fragmented entities using our specialized dispatch logic.
        self._merge_entities_with_dispatch()
        # 2. Update the original relation references to point to the new, merged entities.
        updated_re = self._update_re_annotations()
        
        # Step 3: Run strict adjacency recovery (the forward pass)
        adjacency_recovered = self._recover_strict_adjacency_relations(updated_re)
        
        # Combine all relations found so far
        relations_before_conjunction = updated_re + adjacency_recovered

        # Step 4: Run conjunction recovery (the backward pass) ---
        conjunction_recovered = self._recover_alcohol_drug_conjunction(relations_before_conjunction)

        # Step 5: Combine all relations, remove duplicates, and sort
        all_relations = relations_before_conjunction + conjunction_recovered

        # Step 6: If enabled, remove relations Ti-Tj where i > j
        if self.remove_backward_relations:
            initial_count = len(all_relations)
            # This list comprehension keeps only relations where head_id < tail_id
            all_relations = [
                rel for rel in all_relations
                if int(rel[-2][1:]) < int(rel[-1][1:])
            ]
            if initial_count > len(all_relations):
                print(f"INFO: Repair work complete. Removed {initial_count - len(all_relations)} backward relation(s).")



        seen = set()
        unique_relations = []
        for rel in all_relations:
            # Use the core tuple (name, head, tail) to check for uniqueness
            core_rel = tuple(rel[-3:]) if len(rel) > 2 else tuple(rel)
            if core_rel not in seen:
                unique_relations.append(rel)
                seen.add(core_rel)
        
        self.final_re = sorted(unique_relations, key=lambda x: int(x[-2][1:])) # Sort by head T-ID
        # print("Processing complete.")

    def write_re_to_ann_file(self, re_filename):
        """
        Writes the final, processed annotations to a .ann file.

        Args:
            filename (str): The path for the output .ann file.
        """
        # print(f"Writing final annotations to '{filename}'...")
        try:
            with open(re_filename, 'w', encoding='utf-8') as f:
                # Write relation annotations
                for i, relation_tuple in enumerate(self.final_re):
                    r_id = f"R{i + 1}"
                    rel_name, head_ref, tail_ref = relation_tuple[-3:]
                    f.write(f"{r_id}\t{rel_name} Arg1:{head_ref} Arg2:{tail_ref}\n")
                print(f"File written successfully: {re_filename}")
        except IOError as e:
            print(f"Error writing to ner file: {e}")


    def write_ner_to_ann_file(self, ner_filename):
        """
        Writes the final, processed annotations to a .ann file.

        Args:
            filename (str): The path for the output .ann file.
        """
        # print(f"Writing final annotations to '{filename}'...")
        try:
            with open(ner_filename, 'w', encoding='utf-8') as f:
                # Write entity annotations
                for i, entity_tuple in enumerate(self.merged_ner):
                    t_id = f"T{i + 1}"
                    text, label, (start, end) = entity_tuple
                    f.write(f"{t_id}\t{label} {start} {end}\t{text}\n")

            print(f"File written successfully: {ner_filename}")
        except IOError as e:
            print(f"Error writing to ner file: {e}")

    def display_validation_output(self, context_window=30):
        """
        Prints a rich, formatted output to the console for visually validating
        the final NER and RE results against the original text.

        Args:
            context_window (int): Number of characters to show on each side of an entity.
        """
        print("\n" + "="*40)
        print("   VISUAL VALIDATION REPORT")
        print("="*40)

        # --- NER Validation ---
        print("\n--- [Final NER Annotations] ---\n")
        for i, entity in enumerate(self.merged_ner):
            t_id = f"T{i + 1}"
            label = entity[1]
            highlighted_context = self._get_context_and_highlight(entity, context_window)
            print(f"{t_id.ljust(5)} ({label}): {highlighted_context}")


        # --- RE Validation ---
        print("\n--- [Final RE Annotations] ---\n")
        if not self.final_re:
            print("No relations to display.")


        
        for i, relation in enumerate(self.final_re):
            rel_name, head_ref, tail_ref = relation
            
            # Find the head and tail entities in our final merged list
            head_idx = int(head_ref[1:]) - 1
            tail_idx = int(tail_ref[1:]) - 1
            head_entity = self.merged_ner[head_idx]
            tail_entity = self.merged_ner[tail_idx]
            
            # Get their highlighted contexts
            head_context = self._get_context_and_highlight(head_entity, context_window)
            tail_context = self._get_context_and_highlight(tail_entity, context_window)

            print(f"R{i+1}: {rel_name}")
            print(f"  {head_ref.ljust(5)} ({head_entity[1]}): {head_context}")
            print("          |----------->")
            print(f"  {tail_ref.ljust(5)} ({tail_entity[1]}): {tail_context}")
            print("-" * 30)

    # --- Internal "Private" Methods ---
    def _get_context_and_highlight(self, entity, window):
        """A helper to extract and format a contextual snippet for a given entity."""
        start_span, end_span = entity[2]
        
        # Define the boundaries of our context window, clamping to the text's limits
        context_start = max(0, start_span - window)
        context_end = min(len(self.raw_text), end_span + window)
        
        # Extract the text parts relative to the full raw_text
        before_text = self.raw_text[context_start:start_span]
        entity_text = self.raw_text[start_span:end_span]
        after_text = self.raw_text[end_span:context_end]
        
        # Clean up newlines for cleaner one-line printing
        before_text = before_text.replace('\n', ' ')
        entity_text = entity_text.replace('\n', ' ')
        after_text = after_text.replace('\n', ' ')

        return f"\t ... {before_text}^^^{entity_text}^^^{after_text} ..."

    def _merge_entities_with_dispatch(self):
        """
        The main merging controller. It iterates through entities and dispatches
        them to specialized handlers or a generic one. Populates self.merged_ner
        and self.index_map.
        """
        i = 0
        while i < len(self.raw_ner):
            current_entity = self.raw_ner[i]
            next_entity = self.raw_ner[i + 1] if i + 1 < len(self.raw_ner) else None
            label = current_entity[1]

            if label == 'Abuse':
                final_entity, consumed = self._handle_abuse_merge(current_entity, next_entity)
            elif label == 'Alcohol_use':
                final_entity, consumed = self._handle_alcohol_merge(current_entity, next_entity)
            else: # Generic fallback for all other types
                final_entity, consumed = self._handle_generic_merge(i)

            self.merged_ner.append(final_entity)

            new_index = len(self.merged_ner) - 1
            for k in range(i, i + consumed):
                self.index_map[k] = new_index
            i += consumed

    @staticmethod
    def _handle_abuse_merge(current, next_):
        """Specialized rule for merging 'Fear' and 'or Ex' for 'Abuse' type."""
        if (next_ and next_[1] == 'Abuse' and 
            current[0].strip().lower() == 'fear' and 
            next_[0].strip().lower() == 'or ex'):
            
            # Reconstruct the full, canonical phrase and calculate its new span
            text = "Fear of Current or Ex - Partner"
            start = current[2][0]
            end = start + len(text)
            return (text, 'Abuse', (start, end)), 2
        return current, 1 # If not this specific case, don't merge

    @staticmethod
    def _handle_alcohol_merge(current, next_):
        """Specialized rule for merging 'drink' and 'alcohol' for 'Alcohol_use' type."""
        if (next_ and next_[1] == 'Alcohol_use' and
            current[0].strip().lower() == 'drink' and 
            next_[0].strip().lower() == 'alcohol'):
            
            # Simply concatenate the text and expand the span
            text = f"{current[0]} {next_[0]}"
            start = current[2][0]
            end = next_[2][1]
            return (text, 'Alcohol_use', (start, end)), 2
        return current, 1 # If not this case, don't merge

    def _handle_generic_merge(self, start_index):
        """Generic proximity-based merging for any entity type not specially handled."""
        current_entity = self.raw_ner[start_index]
        label = current_entity[1]
        merged_text = current_entity[0]
        merged_span_end = current_entity[2][1]
        consumed_count = 1
        
        j = start_index + 1
        while j < len(self.raw_ner):
            next_entity = self.raw_ner[j]
            is_same_label = (next_entity[1] == label)
            is_close = (next_entity[2][0] - merged_span_end) <= self.proximity_threshold
            
            if is_same_label and is_close:
                merged_text += " " + next_entity[0]
                merged_span_end = next_entity[2][1]
                consumed_count += 1
                j += 1
            else:
                break
        self.patched_ner.append((merged_text, label, (current_entity[2][0], merged_span_end)))
        final_entity = (merged_text, label, (current_entity[2][0], merged_span_end))
        return final_entity, consumed_count

    # --- NEW: Specific Conjunction Handler ---

    def _recover_alcohol_drug_conjunction(self, current_relations):
        """
        Recovers a missed link for 'Alcohol_use' ONLY when it's part of an
        'Alcohol and drugs: No' pattern. This is a targeted, surgical fix.
        """
        newly_recovered = []
        # Create a set for fast lookup of existing relation indices
        existing_indices = {(int(r[-2][1:]) - 1, int(r[-1][1:]) - 1) for r in current_relations}
        
        # We trigger this logic only off existing 'Drug_use' relations
        for rel in current_relations:
            rel_name, head_ref, tail_ref = rel[-3:]

            # The trigger: an existing relation from Drug_use to a Status
            if rel_name == 'Drug_use-Substance_use_status':
                head_idx = int(head_ref[1:]) - 1
                tail_idx = int(tail_ref[1:]) - 1

                # CONSTRAINT 1: Is there a preceding entity?
                if head_idx > 0:
                    preceding_entity = self.merged_ner[head_idx - 1]
                    
                    # CONSTRAINT 2: Is that preceding entity 'Alcohol_use'?
                    if preceding_entity[1] == 'Alcohol_use':
                        
                        # CONSTRAINT 3: Check if the new relation (Alcohol -> Status) is valid and doesn't already exist
                        if (preceding_entity[1], self.merged_ner[tail_idx][1]) in self.schema_set and \
                           (head_idx - 1, tail_idx) not in existing_indices:
                            
                            print("INFO: Found 'Alcohol and Drug' conjunction pattern. Patching...")
                            # If all checks pass, create the new relation for Alcohol_use
                            new_rel = ('Alcohol_use-Substance_use_status', f"T{head_idx}", f"T{tail_idx + 1}")
                            newly_recovered.append(new_rel)
                            existing_indices.add((head_idx - 1, tail_idx))

        return newly_recovered

    def _update_re_annotations(self):
        """
        Updates existing relation references based on the index map created
        during the merge process.
        
        Returns:
            list: The list of RE annotations with updated 'T' references.
        """
        updated_re = []
        seen = set()
        for rel in self.raw_re:
            old_head_idx = int(rel[1][1:]) - 1
            old_tail_idx = int(rel[2][1:]) - 1
            new_head_idx = self.index_map.get(old_head_idx)
            new_tail_idx = self.index_map.get(old_tail_idx)
            
            if new_head_idx is not None and new_tail_idx is not None and new_head_idx != new_tail_idx:
                new_rel = (rel[0], f"T{new_head_idx + 1}", f"T{new_tail_idx + 1}")
                if new_rel not in seen:
                    updated_re.append(new_rel)
                    seen.add(new_rel)
        return updated_re
        
    def _recover_strict_adjacency_relations(self, existing_relations):
        """
        Recovers missing relations by checking for valid adjacent entity pairs
        in the newly merged NER list. Appends results to self.final_re.
        
        Args:
            existing_relations (list): The list of relations after index updating.
        """
        recovered = []
        existing_indices = { (int(r[1][1:]) - 1, int(r[2][1:]) - 1) for r in existing_relations }
        
        for i in range(len(self.merged_ner) - 1):
            head, tail = self.merged_ner[i], self.merged_ner[i + 1]
            head_label, tail_label = head[1], tail[1]
            
            if (head_label, tail_label) in self.schema_set and (i, i + 1) not in existing_indices:
                rel_name = f"{head_label}-{tail_label}"
                recovered.append((rel_name, f"T{i + 1}", f"T{i + 2}"))
        
        # Combine existing and recovered relations and sort for consistency
        self.existing_re = existing_relations
        self.recovered_re = recovered
        self.final_re = sorted(existing_relations + recovered, key=lambda x: int(x[1][1:]))
        return recovered


def get_text_annotations(note_id, BRAT_ANN_DIR, BRAT_RE_ANN_DIR, RAW_TEXT_DIR):

    from NLPreprocessing.annotation2BIO import read_annotation_brat
    try:
        text_file = RAW_TEXT_DIR / f"{note_id}.txt"
        ann_file = BRAT_ANN_DIR / f"{note_id}.ann"
        ann_re_file = BRAT_RE_ANN_DIR / f"{note_id}.ann"
    except Exception as e:
        print("Problem loading file:")
        print(str(e), note_id)
        raise

    with open(text_file, 'r') as f:
        text = f.read()
        
    annotations = read_annotation_brat(ann_file)[1]
    re_annotations = read_annotation_brat(ann_re_file)[2]

    return text, annotations, re_annotations

def apply_patch_processor(note_id, BRAT_ANN_DIR, BRAT_RE_ANN_DIR, RAW_TEXT_DIR, NER_OUTPUT_DIR, RE_OUTPUT_DIR, debug = False, remove_backward_relations = True):

    raw_text, ner_annotations, re_annotations = get_text_annotations(note_id, BRAT_ANN_DIR, BRAT_RE_ANN_DIR, RAW_TEXT_DIR)

    # 1. Instantiate the processor with the data
    processor = PostProcessor(ner_annotations, re_annotations, raw_text, remove_backward_relations = remove_backward_relations)

    # 2. Run the entire pipeline
    processor.process()

    if debug:
        processor.display_validation_output()

        # 3. Print the results to the console to verify
        print("\n--- Final Results ---")
        print("Cleaned NER Annotations:")
        for start, entity in enumerate(processor.merged_ner, start=1):
            if entity in processor.patched_ner:
                print(f"T{start}:  {entity}")
            else:
                print(f"T{start} (patched): {entity}")
        
        print("\nFinal RE Annotations (with recovered relations):")
        for start, relation in enumerate(processor.final_re, start=1):
            if relation in processor.existing_re:
                print(f"R{start} (existing): {relation}")
            else:
                print(f"R{start} (recovered): {relation}")
        
    # 4. Save the final output to a file
    

    ner_filename = NER_OUTPUT_DIR / f"{note_id}.ann"
    re_filename = RE_OUTPUT_DIR / f"{note_id}.ann"

    # print(f"Saving cleaned annotations to {output_ann_file}...\n")
    processor.write_ner_to_ann_file(ner_filename)
    processor.write_re_to_ann_file(re_filename)

    return processor


if __name__ == "__main__":
    test_note_id = 5474145401 

    BRAT_ANN_DIR = Path("/blue/yonghui.wu/aman.pathak/projects/IDR_SDoH/SDOH_BASEPATH/brat/")
    BRAT_RE_ANN_DIR = Path("/blue/yonghui.wu/aman.pathak/projects/IDR_SDoH/SDOH_BASEPATH/brat_re/")
    RAW_TEXT_DIR = Path("/blue/yonghui.wu/aman.pathak/projects/IDR_SDoH/SDOH_BASEPATH/raw_text/")
    NER_OUTPUT_DIR = Path("/blue/yonghui.wu/aman.pathak/projects/IDR_SDoH/SDOH_BASEPATH/cleaned_brat/")
    RE_OUTPUT_DIR = Path("/blue/yonghui.wu/aman.pathak/projects/IDR_SDoH/SDOH_BASEPATH/cleaned_brat_re/")

    processor = apply_patch_processor(test_note_id, BRAT_ANN_DIR, BRAT_RE_ANN_DIR, RAW_TEXT_DIR,
     NER_OUTPUT_DIR = Path("/orange/yonghui.wu/dparedespardo/Noah_SODA_LLM/notebooks/test/"), 
     RE_OUTPUT_DIR = Path("/orange/yonghui.wu/dparedespardo/Noah_SODA_LLM/notebooks/"), debug=False)
    print("File saved!")

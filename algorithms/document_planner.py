
from utils.llm_client import OpenAIClient
import logging
from typing import List, Dict, Any, Optional,Tuple
import json
from .data_structure import Block
def associate_blocks_llm(exp_table: Block, all_blocks: List[Block], llm: OpenAIClient) -> List[int]:
    candidate_blocks = [b for b in all_blocks if b.id != exp_table.id and abs(b.page - exp_table.page) <= 1]
    if len(candidate_blocks) > 30:
        candidate_blocks = candidate_blocks[:30]
    candidate_info = []
    for cb in candidate_blocks:
        info = {
            "id": cb.id,
            "type": cb.type,
            "page": cb.page,
            "content_preview": (cb.content if cb.content else (cb.image_caption if cb.image_caption else "image")),
        }
        candidate_info.append(info)
    prompt = f"""You are given an experimental data table and a list of other blocks (text, images) from the same paper. Your task is to select which blocks are **relevant** to understanding the experiments in this table. Relevance includes:
- Blocks that provide explanations of table footnotes or abbreviations.
- Blocks that describe the general experimental procedure.
- Blocks that mention the same entry/run numbers as the table.
- Figure captions that show molecular structures or GPC curves related to these experiments.

Table ID: {exp_table.id}
Table title: {exp_table.table_title}
Table body (first 500 chars): {exp_table.table_body if exp_table.table_body else exp_table.content}

Candidate blocks:
{json.dumps(candidate_info, indent=2)}

Output a JSON list of block IDs that are relevant. Only include IDs from the candidate list. Example: [3, 7, 12]
"""
    response, usage = llm.call_with_usage(prompt, temperature=0.0, max_tokens=500)
    print(response)
    related_ids = llm.extract_json_from_response(response)
    if not isinstance(related_ids, list):
        logging.warning(f"LLM returned invalid association for table {exp_table.id}, using empty list")
        related_ids = []
    related_ids_new = [idx for idx in related_ids if idx != exp_table.id]
    logging.info(f"LLM returned {related_ids_new} associations for table {exp_table.id}")
    return related_ids_new
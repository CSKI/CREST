import json
from typing import List, Dict, Any, Optional
from .data_structure import Block,build_full_table_context, build_full_context
from utils.llm_client import OpenAIClient
import logging
import copy
import os

logger = logging.getLogger(__name__)

def identify_experiment_tables_llm(blocks: List[Block], 
                                   llm_client: OpenAIClient) -> List[Block]:
    """
    使用 LLM 从所有表格块中筛选出实验数据表格。
    如果论文中没有表格，可以扩展为识别段落实验。
    """
    # 过滤出所有 table 类型的块
    table_blocks = [b for b in blocks if b.type == "table"]
    if not table_blocks:
        return []

    # 构建 prompt
    tables_info = []
    for tb in table_blocks:
        tables_info.append({
            "id": tb.id,
            "title": tb.table_title,
            "first_few_rows": tb.table_body[:300] if tb.table_body else tb.content[:300]
        })

    prompt = f"""
You are a chemistry data extraction assistant specialized in MAO-activated metal catalysts (e.g., olefin polymerization, COC, etc.).

The following are tables extracted from a chemistry paper. Identify which tables contain **experimental catalytic data**, including but not limited to:
- Polymerization/catalysis results (activity, yield, conversion, turnover number/frequency)
- Reaction conditions (temperature, pressure, time, solvent, monomer, Al/M ratio, cocatalyst amount)
- Catalyst performance metrics (molecular weight, PDI, melting point, etc.) **even if they appear together with characterization data**

Exclude ONLY tables that are **purely**:
- Supporting information that is fully about compound characterization (NMR, MS, IR, XRD, crystal data)
- Reference tables (e.g., literature comparison without new experimental data)
- Purely physical property tables (melting points of pure compounds without catalysis context)

If a table contains BOTH catalytic performance data AND some characterization columns, include it.

Tables:
{json.dumps(tables_info, indent=2)}

Output a JSON list of table IDs that are experimental data tables. Example: [2, 5]
"""
    response, use_age = llm_client.call_with_usage(prompt, temperature=0.0)
    exp_table_ids = llm_client.extract_json_from_response(response)
    if not isinstance(exp_table_ids, list):
        logger.warning("LLM did not return a list, falling back to all tables")
        exp_table_ids = [tb.id for tb in table_blocks]
    print(exp_table_ids)
    return [tb for tb in table_blocks if tb.id in exp_table_ids]

def add_evidence_to_records(
    records: List[Dict],           # 已提取的记录（包含原始值）
    exp_table: Block,
    related_blocks: List[Block],
    llm_client: OpenAIClient,
) -> List[Dict]:
    """
    为已提取的记录添加 evidence（来源 block_id）。
    输出格式：每个记录中，每个字段的值被替换为
    对应的 block_id（字符串）。
    原始值会被丢弃（因为已经通过第一个 LLM 获得）。
    最终返回的格式：
    [
        {"field1": "block_id_1", "field2": "block_id_2", "additional_items": [...]},
        ...
    ]
    其中 additional_items 中的每个 item 也变成 {"item_name": ..., "evidence": "block_id"}
    """
    if not records:
        return records

    full_context, table_text = build_full_table_context(exp_table, related_blocks)

    # 构建一个简化的 prompt，要求 LLM 只输出每个字段对应的最清晰 block_id
    prompt = f"""
You are given extracted experimental records and the original paper context.
Your task: For each field in each record, identify the single most relevant block_id where that information originates.
The block_id can be from the table (id: {exp_table.id}) or from other blocks (ids: {[b.id for b in related_blocks if b.id != exp_table.id]}).
You do NOT need to output the original values; only output the block_id for each field.

Original context (use this to find evidence):
{table_text}
{full_context}

Extracted records (values shown for reference, but you will not output them):
{json.dumps(records, indent=2, ensure_ascii=False)}

Output a JSON list of the same length. Each element is an object where:
- Keys are the same field names as in the input record (including "additional_items" if present).
- Values are the block_id (string) that best supports that field.

Example output format:
[
    {{
        "catalyst": "5",
        "temperature": "5",
        "yield": "1", 
        "XXX": "", ## no evidence for this field
        ....
    }},
    .....more records
]

Only output the JSON list, no other text.
"""
    response, usage = llm_client.call_with_usage(prompt)
    try:
        evidence_map = llm_client.extract_json_from_response(response)
        if not isinstance(evidence_map, list) or len(evidence_map) != len(records):
            logger.error("Evidence annotation failed, returning records without evidence")
            return []
        return evidence_map
    except Exception as e:
        logger.error(f"Error extracting evidence: {e}")
        return []

def build_none_response(records: List[Dict]) -> List[Dict]:
    records = copy.deepcopy(records)
    for record in records:
        for key in record:
            record[key] = None
    return records

def add_evidence_to_records_from_paper(
    records: List[Dict],
    related_blocks: List[Block],
    schema: Dict,
    llm_client: OpenAIClient,
) -> List[Dict]:
    if not records:
        return records

    full_context = build_full_context(related_blocks)
    block_ids = [b.id for b in related_blocks]
    
    schema_desc = ""
    if schema:
        schema_desc = f"Field definitions: {json.dumps(schema, ensure_ascii=False)}\n"

    prompt = f"""
You are given extracted experimental records and the original paper context.
Your task: For each field in each record, find the most relevant block_id (from available IDs) that supports that field's value.
Available block IDs: {block_ids}
{schema_desc}
Original context:
{full_context}

Extracted records (values shown for reference, but you will output only block_ids):
{json.dumps(records, indent=2, ensure_ascii=False)}

Output a JSON list of the same length. Each element is an object where:
- Keys are the same field names as in the input record (including "additional_items" if present).
- Values are the block_id (as a string) that best supports that field. If no supporting block exists, use an empty string "".

Example:
[
    {{"catalyst": "5", "temperature": "37", "yield": "80", "solvent": ""}},
    ...
]

Only output the JSON list, no other text.
"""
    response, usage = llm_client.call_with_usage(prompt)
    try:
        evidence_map = llm_client.extract_json_from_response(response)
        if not isinstance(evidence_map, list) or len(evidence_map) != len(records):
            logger.error("Evidence annotation failed, returning empty evidence")
            return build_none_response(records)
        return evidence_map
    except Exception as e:
        logger.error(f"Error extracting evidence: {e}")
        return build_none_response(records)
    
    
def add_confidence_to_records_from_paper(
    records: List[Dict],           # 已提取的记录（包含原始值）
    related_blocks: List[Block],
    schema: Dict,
    llm_client: OpenAIClient,

) -> List[Dict]:
    """
    为已提取的记录添加 evidence（来源 block_id）。
    输出格式：每个记录中，每个字段的值被替换为
    对应的 block_id（字符串）。
    原始值会被丢弃（因为已经通过第一个 LLM 获得）。
    最终返回的格式：
    [
        {"field1": 0.1, "field2": 0.9},
        ...
    ]
    其中 additional_items 中的每个 item 也变成 {"item_name": ..., "evidence": "block_id"}
    """
    if not records:
        return records

    full_context = build_full_context(related_blocks)
    schema_desc = ""
    if schema:
        schema_desc = f"Field definitions: {json.dumps(schema, ensure_ascii=False)}\n"

    # 构建一个简化的 prompt，要求 LLM 只输出每个字段对应的最清晰 block_id
    prompt = f"""
You are an expert in evaluating experimental data extraction.  
Given the schema definition, the original paper context, and a set of extracted records,  
your task is to assign a **confidence score** (0.0 to 1.0) to each extracted field value,  
indicating how well it matches the information in the context.

**Scoring criteria:**
- 1.0 = Exact match, clearly stated in context.
- 0.8-0.9 = Slight paraphrase or unit variation, but clearly correct.
- 0.5-0.7 = Partially correct (e.g., missing a modifier, ambiguous).
- 0.1-0.4 = Likely incorrect, but some weak connection.
- 0.0 = No evidence or clearly contradictory.

**Schema:**
{schema_desc}

**Context:**
{full_context}

**Extracted records (for reference only):**
{json.dumps(records, indent=2, ensure_ascii=False)}

Output a JSON list of the same length as the records. Each element is an object with the same field names as the input record, and the value is a float confidence score.

Example output for a single record:
[
    {{"catalyst": 1.0, "temperature": 0.6, "yield": 0.9}},
    ...more records...
]

Only output the JSON list. No extra text.
"""
    response, usage = llm_client.call_with_usage(prompt)
    try:
        confidence_map = llm_client.extract_json_from_response(response)
        if not isinstance(confidence_map, list) or len(confidence_map) != len(records):
            logger.error("Evidence annotation failed, returning records without evidence")
            return build_none_response(records)
        return confidence_map
    except Exception as e:
        logger.error(f"Error extracting evidence: {e}")
        return build_none_response(records)

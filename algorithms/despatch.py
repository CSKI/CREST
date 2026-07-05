import json
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data_structure import Block,TokenTracker, build_full_table_context
from utils import OpenAIClient
from bs4 import BeautifulSoup
import copy
import time
logger = logging.getLogger(__name__)

def normalize_record(rec: Dict) -> Dict:
    """Convert additional_items to marked fields."""
    if "additional_items" in rec:
        for item in rec["additional_items"]:
            if not isinstance(item, dict):
                continue
            field_name = item.get("item_name")
            if field_name:
                rec[field_name] = {
                "value": item.get("value", ""),
                "definition": item.get("define", ""),
                "suggested_type": item.get("suggested_data_type", "string"),
                "_is_extra": True
                }
        del rec["additional_items"]
    new_rec = {}
    for k, v in rec.items():
        if isinstance(v, dict) and "_is_extra" in v:
            if v["value"] is None or v["value"] == "":
                continue
            new_rec[k] = v
        else:
            if v is None or v == "":
                continue       
            new_rec[k] = v
    return new_rec

def extract_table_rows_to_list(html_content, table_index=0):
    """
    从 HTML 中提取指定表格的所有行内容，返回列表的列表。
    
    参数:
        html_content (str): HTML 字符串
        table_index (int): 如果 HTML 中有多个表格，指定要提取的表格索引（默认 0，即第一个表格）
    
    返回:
        list: 每个元素是一个列表，对应一行中所有单元格（th 或 td）的文本内容。
              示例: [['Name', 'Age'], ['Alice', '25'], ['Bob', '30']]
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    
    if not tables:
        raise ValueError("未找到任何表格")
    if table_index >= len(tables):
        raise IndexError(f"表格索引 {table_index} 超出范围，共 {len(tables)} 个表格")
    
    target_table = tables[table_index]
    rows = target_table.find_all('tr')
    
    result = []
    for row in rows:
        # 获取该行中所有的 th 和 td 单元格
        cells = row.find_all(['th', 'td'])
        row_content = [cell.get_text(strip=True) for cell in cells]
        result.append(row_content)
    
    return result

def list_to_markdown(table_data):
    """
    将二维列表转换为 Markdown 表格。

    参数:
        table_data (list of list): 表格数据，第一行通常为表头，后续为数据行。

    返回:
        str: Markdown 格式的表格字符串。
    """
    if not table_data:
        return ""

    # 确定最大列数（处理不规则行）
    max_cols = max(len(row) for row in table_data) if table_data else 0
    if max_cols == 0:
        return ""

    # 补全不足列数为空字符串
    normalized = [row + [""] * (max_cols - len(row)) for row in table_data]

    # 构建每一行
    markdown_rows = []
    for i, row in enumerate(normalized):
        # 单元格内容转义管道符和换行符（可选）
        escaped = [cell.replace("|", "\\|").replace("\n", "<br>") for cell in row]
        markdown_rows.append("| " + " | ".join(escaped) + " |")
        # 第一行后添加分隔行
        if i == 0:
            separator = "| " + " | ".join(["---"] * max_cols) + " |"
            markdown_rows.append(separator)

    return "\n".join(markdown_rows)

def process_single_table(
    exp_table: Block,
    all_blocks: List[Block],
    schema: List[Dict],
    llm_client: OpenAIClient,
    tracker: TokenTracker,
    batch_size: int = 10
) -> List[Dict]:
    """处理一个实验表格，返回提取的所有记录"""
    logger.info(f"Processing table {exp_table.id}")
    # 1. 解析表格为二维列表
    try:
        rows = extract_table_rows_to_list(exp_table.table_body)
    except Exception as e:
        logger.error(f"Failed to parse table {exp_table.id}: {e}")
        return []
    if not rows:
        return []
    # 2. 识别表头行
    header_indices = identify_table_headers(rows, llm_client, tracker)
    if not header_indices:
        # 默认第一行为表头
        header_indices = [0]
    header_rows = [rows[i] for i in header_indices]
    body_rows = [rows[i] for i in range(len(rows)) if i not in header_indices]

    # 6. 分批提取记录
    all_records = []
    for start in range(0, len(body_rows), batch_size):
        batch = header_rows + body_rows[start:start+batch_size]
        exp_table.table_body = list_to_markdown(batch)
        rest_blocks_len = len(body_rows[start:start+batch_size])

        batch_records = extract_records_with_evidence(
            exp_table, 
            all_blocks, 
            schema, 
            llm_client, 
            tracker,
            rest_blocks_len
        )
        print(batch_records)

        all_records.append(batch_records)
        break
    logger.info(f"Table {exp_table.id}: extracted {len(all_records)} records")
    return all_records

def identify_table_headers(rows: List[List[str]], 
                           llm_client: OpenAIClient, 
                           tracker: TokenTracker) -> List[int]:
    """返回作为表头的行索引列表"""
    if not rows:
        return []
    rows_info = [{"idx": i, "content": row} for i, row in enumerate(rows)]
    prompt = f"""Based on the table rows below, identify which rows are header rows (contain column names). Output a list of row indices (0-based). If no header row, output [].

Rows:
{json.dumps(rows_info, indent=2)}

Output only the JSON list, e.g., [0] or [0,1].
"""
    response, usage = llm_client.call_with_usage(prompt, temperature=0.0)
    if usage:
        tracker.add_usage(usage)
    header_indices = llm_client.extract_json_from_response(response)
    if not isinstance(header_indices, list):
        return []
    return header_indices
   
def identify_experiment_tables_llm(blocks: List[Block], 
                                   llm_client: OpenAIClient,
                                   tracker: TokenTracker) -> List[Block]:
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

    prompt = f"""You are a chemistry data extraction assistant. The following are tables extracted from a chemistry paper. Identify which tables contain experimental data (polymerization/catalysis results, reaction conditions, yields, activities, etc.). Exclude tables that are for supporting information, characterization of compounds (NMR, MS, crystal data), or reference tables.

Tables:
{json.dumps(tables_info, indent=2)}

Output a JSON list of table IDs that are experimental data tables. Example: [2, 5]
"""
    response, use_age = llm_client.call_with_usage(prompt, temperature=0.0)
    if use_age:
        tracker.add_usage(use_age)
    exp_table_ids = llm_client.extract_json_from_response(response)
    if not isinstance(exp_table_ids, list):
        logger.warning("LLM did not return a list, falling back to all tables")
        exp_table_ids = [tb.id for tb in table_blocks]

    return [tb for tb in table_blocks if tb.id in exp_table_ids]

def extract_records_with_evidence(
    exp_table: Block,
    related_blocks: List[Block],
    schema: List[Dict],
    llm_client: OpenAIClient,
    tracker: TokenTracker,
    record_number: int=1
) ->  Dict:
    """
    提取当前批次的实验记录，每个记录包含字段值及 `_evidence` 字段。
    """
    """
    将实验表格及其相关块的内容整合成一个长上下文，调用 LLM 提取每一行实验的参数。
    输出一个 JSON 列表，每个元素是一个实验记录（字典），并附上证据。
    """
    # 构建上下文：按块 ID 顺序排列，每个块标明来源
    full_context, table_text = build_full_table_context(exp_table, related_blocks)
    prompt = f"""
# Role
You are an expert chemical data extraction engineer. Your task is to extract experimental records from a given chemistry paper and format them according to a specified database schema.

# Input Data
- **Table Data**: {table_text}
- **Full Paper Context**: {full_context}
- **Database Schema**: {schema}

# Extraction Rules
1. **Completeness**: Extract **all** experimental records mentioned in the paper (including those in tables, main text, supporting information, etc.). Do not omit any.
2. **Schema Adherence**: For each extracted record, map the values to the fields defined in the `Database Schema` whenever possible.
3. **Exclude `block_id`**: Even if `block_id` appears in the provided `{schema}`, **do not output it** in the extracted JSON. Skip this field entirely.
4. **Omit null/empty values**: If a standard field (other than `block_id`) cannot be extracted, **do not include that field in the output at all**. Only output fields that have a non‑null, non‑empty value.
5. **Handling Non‑standard Items**:
   - If the paper contains experimental attributes that do **not** exist in the provided schema, you must capture them in the `additional_items` array (see output format).
   - For each such item, provide:
     - `item_name`: the original field name (e.g., "catalyst_loading")
     - `define`: a brief definition or description of this field
     - `value`: the extracted value (preserve units and raw text)
     - `suggested_data_type`: e.g., string, number, float, boolean, array, etc.
   - If a non‑standard item has an empty or missing value, **do not include it** in the `additional_items` array.
6. **Multiple Records**: If the paper describes multiple separate experiments (e.g., different reaction conditions, different compounds..), output **one JSON object per experiment** in a JSON array.

# Output Format
Output **only** valid JSON. Use the exact structure below. The `additional_items` array must be included **even if empty**.

```json
[
  {{
    #// Only include standard fields (from schema, except block_id) that have a non‑null, non‑empty value.
    // Example: "reaction_temperature": "120 °C"
    // Do NOT include fields like "yield": null or "solvent": ""
    "additional_items": [
      {{
        "item_name": "string (name of the non‑standard field)",
        "define": "string (definition/description)",
        "value": "string (extracted value, may include units)",
        "suggested_data_type": "string (e.g., number, string, boolean)"
      }}
    ]
  }}
]
"""
    extract_info = {}
    response, usage = llm_client.call_with_usage(prompt, temperature=0.1)
    extract_info['un_parsed'] = response
    extract_info['extract_usage'] = usage
    records = llm_client.extract_json_from_response(response)
    extract_info['parsed'] = records
    if not isinstance(records, list):
        logger.error("LLM did not return a list of records")
        records = [] * record_number
        extract_info['parsed'] = records
        return extract_info
    new_records = []
    for record in records:
        record = normalize_record(record)
        new_records.append(record)
    
    enriched_records = add_evidence_to_records(new_records, exp_table, related_blocks, llm_client, tracker)
    extract_info["records"] = enriched_records
    extract_info["evidence_usage"] = usage
    return extract_info

def add_evidence_to_records(
    records: List[Dict],           # 已提取的记录（包含原始值）
    exp_table: Block,
    related_blocks: List[Block],
    llm_client: OpenAIClient,
    tracker: TokenTracker
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
    response, usage = llm_client.call_with_usage(prompt, temperature=0.0)
    if usage:
        tracker.add_usage(usage)
    
    evidence_map = llm_client.extract_json_from_response(response)
    if not isinstance(evidence_map, list) or len(evidence_map) != len(records):
        logger.error("Evidence annotation failed, returning records without evidence")
        return records


    # 合并原始值和证据（将值替换为 evidence）
    enriched_records = []
    for orig_rec, ev_map in zip(records, evidence_map):
        new_rec = {}
        # 处理普通字段
        for k, v in orig_rec.items():
            # 如果 ev_map 中有该字段的证据，则使用；否则使用 "unknown"
            evidence = ev_map.get(k, "unknown")
            if isinstance(v, dict):
                orig_rec[k]['evidence'] = evidence
            else:
                orig_rec[k] = {"value": v, "evidence": evidence}
        enriched_records.append(orig_rec)
    return enriched_records

def batch_process_table(
    exp_table: Block,
    all_blocks: List[Block],
    schema: List[Dict],
    llm_client: OpenAIClient,
    tracker: TokenTracker,
    batch_size: int = 10,
    inner_max_workers: int = 4   # 内部批次并行度
) -> List[Dict]:
    """处理一个实验表格，内部批次并行提取"""
    logger.info(f"Processing table {exp_table.id}")
    # 1. 解析表格为二维列表
    try:
        rows = extract_table_rows_to_list(exp_table.table_body)
    except Exception as e:
        logger.error(f"Failed to parse table {exp_table.id}: {e}")
        return []
    if not rows:
        return []
    # 2. 识别表头行
    header_indices = identify_table_headers(rows, llm_client, tracker)
    if not header_indices:
        header_indices = [0]
    header_rows = [rows[i] for i in header_indices]
    body_rows = [rows[i] for i in range(len(rows)) if i not in header_indices]

    related_blocks = all_blocks  # 或者根据上下文筛选

    # 4. 准备批次（只包含表体行，表头固定）
    batches = []
    for start in range(0, len(body_rows), batch_size):
        batch_body = body_rows[start:start+batch_size]
        # 每个批次拼接表头 + 当前批次表体
        batch_table = header_rows + batch_body
        batches.append({"exp_data":copy.deepcopy(exp_table), "batch_table": batch_table, "records_num": len(batch_body)})

    # 5. 并行处理每个批次
    # 使用列表预分配保持顺序
    batch_results = [{}] * len(batches)

    def process_one_batch(idx, batch_data_exp):
        """处理单个批次，返回 (idx, records)"""
        # 临时修改 exp_table 的 table_body 为当前批次的 Markdown
        batch_data = batch_data_exp['batch_table']
        exp_table = batch_data_exp['exp_data']
        record_num = batch_data_exp['records_num']
        original_body = exp_table.table_body
        exp_table.table_body = list_to_markdown(batch_data)
        try:
            records = extract_records_with_evidence(
                exp_table, related_blocks, schema, llm_client, tracker,record_num
            )
        finally:
            exp_table.table_body = original_body   # 恢复
        return idx, records

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = {
            executor.submit(process_one_batch, idx, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx, recs = future.result()
            batch_results[idx] = recs

    # 展平结果，保持原顺序
    all_records = []
    for recs in batch_results:
        if recs:
            all_records.append(recs)

    logger.info(f"Table {exp_table.id}: extracted {len(all_records)} records")
    return all_records

class DynamicSchemaExtractor:
    """Extract experiment records from paper text using core fields + additional_items."""
    def __init__(self,
                 llm_client: OpenAIClient,
                 batch_size: int = 10, 
                 schema={}):
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.tracker = TokenTracker()
        self.schema = schema
        
    def single_extract_records(self, block_objects: List[Block]) -> List[Dict]:
        """Extract records from text using LLM."""
        
        # 1. 识别实验表格
        exp_tables = identify_experiment_tables_llm(block_objects, 
                                                    self.llm_client,
                                                    self.tracker)
        if not exp_tables:
            logger.warning("No experiment tables found.")
            return []
        logger.info(f"Found {len(exp_tables)} experiment tables")
        all_records = []
        new_blocks = []
        for block in block_objects:
            if block not in exp_tables:
                new_blocks.append(block)
        
        for et in exp_tables:
            records = process_single_table(et, 
                                           new_blocks,
                                           self.schema,
                                           self.llm_client,
                                           self.tracker, 
                                           batch_size=self.batch_size
                                           )
            
            all_records.append({"block_id": et.id, "records": records})
            break
        return all_records

    def extract_records(self, block_objects: List[Block]):
        self.tracker = TokenTracker()
        """并行处理多个实验表格，提取记录。"""
        # 1. 识别实验表格
        exp_tables = identify_experiment_tables_llm(block_objects,
                                                    self.llm_client,
                                                    self.tracker)
        if not exp_tables:
            logger.warning("No experiment tables found.")
            return []
        logger.info(f"Found {len(exp_tables)} experiment tables")
        
        new_blocks = []
        for block in block_objects:
            if block not in exp_tables:
                new_blocks.append(block)
        
        # 2. 并行处理每个表格
        all_results = []   # 每个元素为 {"block_id": id, "records": [...]}
        with ThreadPoolExecutor(max_workers=len(exp_tables)) as executor:
            future_to_table = {
                executor.submit(
                    batch_process_table,
                    et,
                    new_blocks,
                    self.schema,
                    self.llm_client,
                    self.tracker,
                    self.batch_size
                ): et
                for et in exp_tables
            }
            for future in as_completed(future_to_table):
                et = future_to_table[future]
                try:
                    records = future.result()
                    all_results.append({"block_id": et.id, "records": records})
                except Exception as e:
                    logger.error(f"Table {et.id} processing failed: {e}")
                    all_results.append({"block_id": et.id, "records": [], "error": str(e)})

        
        return all_results
    


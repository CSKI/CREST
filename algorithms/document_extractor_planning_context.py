import json
import copy
from typing import List, Dict, Any, Optional
from utils.llm_client import OpenAIClient

logger = logging.getLogger(__name__)

class ChunkExtractor:
    """
    基于共享基础上下文 A，独立处理每个 chunk 的提取和证据定位。
    A 包括：表格HTML、相关块文本、行数估计结果（可选）。
    """
    def __init__(self, llm_client: OpenAIClient, schema: List[Dict]):
        self.client = llm_client
        self.schema = schema

    def build_base_context(self, table_html: str, context_blocks: List[Any], total_rows: int = None) -> List[Dict]:
        """
        构建基础上下文 A（消息列表格式）。
        如果 total_rows 未知，可以调用 _estimate_rows 来获得。
        """
        if total_rows is None:
            total_rows = self._estimate_rows(table_html, context_blocks)
        
        context_text = "\n\n".join([block.content for block in context_blocks])
        base_messages = [
            {"role": "system", "content": "You are an expert in chemical data extraction. You will be given a table and its context."},
            {"role": "user", "content": f"""
Here is the experimental table (HTML format) and its relevant context blocks.
We will later extract data rows from the table.

Table HTML:
{table_html}

Relevant context (procedures, methods, footnotes, etc.):
{context_text}

The table has {total_rows} data rows (row 1 is the first data row after the header, do not count the header row).
Remember this information for subsequent extractions.
"""}
        ]
        # 可选：添加模型确认消息（但为了简洁，这里不等待模型响应）
        return base_messages

    def extract_chunk(self, base_messages: List[Dict], start_row: int, end_row: int) -> List[Dict]:
        """
        基于基础上下文 A，提取指定行范围的记录（原始值）。
        返回记录列表，每条记录是一个字典（字段名 → 论文中的原始值）。
        """
        messages = copy.deepcopy(base_messages)
        user_content = f"""
Now extract the experimental records for rows {start_row} to {end_row} (inclusive) from the table.
Row numbering: 1 is the first data row after the header.

Use the following schema:
{json.dumps(self.schema, indent=2)}

Output a JSON array of objects, each object corresponds to one row.
If a field's value is empty, omit it. Include non‑schema fields in "additional_items" array.
Only output the JSON array, no extra text.
"""
        messages.append({"role": "user", "content": user_content})
        response = self.client.call_with_history(messages, temperature=0.1)
        parsed = self.client.extract_json_from_response(response)
        if isinstance(parsed, list):
            # 标准化记录（处理 additional_items 等）
            return [self._normalize_record(rec) for rec in parsed if isinstance(rec, dict)]
        logger.error(f"Failed to extract chunk rows {start_row}-{end_row}")
        return []

    def add_evidence(self, base_messages: List[Dict], records: List[Dict]) -> List[Dict]:
        """
        基于基础上下文 A + 已提取的记录 B，为每个字段添加证据 block_id。
        返回的新记录中，字段的值变为 block_id 字符串（或 ""）。
        """
        if not records:
            return records
        messages = copy.deepcopy(base_messages)
        user_content = f"""
For the following extracted records, add the source block_id for each field.
Use the original context (the table and blocks provided earlier) to locate the evidence.
For each record, output a JSON object with the same keys, but values replaced by the block_id string (or "" if not found).
For "additional_items", output an object with "item_name" and "evidence".

Records:
{json.dumps(records, indent=2, ensure_ascii=False)}

Output a JSON list of the same length as the records.
Only output the JSON list.
"""
        messages.append({"role": "user", "content": user_content})
        response = self.client.call_with_history(messages, temperature=0.0)
        evidence_map = self.client.extract_json_from_response(response)
        if isinstance(evidence_map, list) and len(evidence_map) == len(records):
            final_records = []
            for ev, orig in zip(evidence_map, records):
                new_rec = {}
                # 确保所有原始字段都出现在 evidence 中（缺失的补 "")
                for field in orig.keys():
                    new_rec[field] = ev.get(field, "")
                final_records.append(new_rec)
            return final_records
        logger.error("Evidence annotation failed, returning records without evidence")
        return records

    def _estimate_rows(self, table_html: str, context_blocks: List[Any]) -> int:
        """内部方法：通过 LLM 估计表格行数（不保存到历史，只用于构建 base_messages）"""
        context_text = "\n\n".join([block.content for block in context_blocks])
        prompt = f"""
Given the table and context, estimate how many data rows (experiment records) are in the table.
Do NOT count the header row.

Table HTML:
{table_html}

Context:
{context_text}

Return a JSON object: {{"total_records": integer}}.
Only output the JSON object.
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.call_with_history(messages, temperature=0.0)
        try:
            data = self.client.extract_json_from_response(response)
            if isinstance(data, dict) and "total_records" in data:
                return int(data["total_records"])
        except:
            pass
        # 降级：尝试从 HTML 中解析行数（简单 heuristic）
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(table_html, 'html.parser')
            rows = soup.find_all('tr')
            # 假设第一行是表头
            return max(0, len(rows) - 1)
        except:
            return 0

    def _normalize_record(self, rec: Dict) -> Dict:
        """将 additional_items 转换为标记字段（保留原始值）"""
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
        return rec
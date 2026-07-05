import json
import logging
from typing import List, Dict, Any, Optional

from utils.llm_client import OpenAIClient

logger = logging.getLogger(__name__)

class DocumentExtractorCompress:
    """Extract experiment records from paper text using core fields + additional_items."""
    def __init__(self, llm_client: OpenAIClient, schema):
        self.client = llm_client

        self.schema_str = schema

    def extract(self, paper_text: str) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(paper_text)
        try:
            response = self.client.call(
                prompt
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return []
        if not response:
            logger.warning("LLM returned empty response")
            return []

        logger.debug(f"LLM response length: {len(response)} characters")

        # 解析列式 JSON
        columnar_data = self._parse_response(response)
        if not columnar_data:
            logger.error("Failed to parse LLM response into JSON")
            logger.debug(f"Failed response snippet: {response[:500]}...")
            return []

        # 转换为列表形式
        records = self._convert_columnar_to_records(columnar_data)

        logger.info(f"Extracted {len(records)} experiment(s)")
        return records

    def _build_prompt(self, paper_text: str) -> str:
        return f"""
Hello, as a chemical data extraction engineer, there is currently a paper in the field of chemistry here. Please extract the experimental records from it. I hope that the extracted experimental items can meet my database standards.

## Paper data:
{paper_text}

## Database standards:
{self.schema_str}

## Note:
1. Do not omit data.
2. If there are additional items or experimental items that do not exist in the database standard items, please also list them as schema format.
3. Identify each distinct experiment (e.g., based on table rows, sequential paragraphs, or sections). Keep the order of experiments as they appear in the paper.
4. **Output must be a compact JSON (no spaces, no newlines) following the structure below.**

## Output format:
'''json
[
  {{
    // Standard fields: each field's value is an array of values for all experiments, in order.
    "field_name_1": [value1_exp1, value1_exp2, ...],
    "field_name_2": [value2_exp1, value2_exp2, ...],
    // ... other standard fields

    // Extra fields (not in standard schema):
    "extra": [
      {{
        "name": "extra_field_name",
        "define": "brief definition",
        "type": "suggested data type (string, float, integer, etc.)",
        "value": [value1_exp1, value1_exp2, ...]
      }},
      // ... more extra fields if needed
    ]
  }}
  ... more experiments if needed
]
'''

**Important:**
- All arrays (both standard field arrays and extra `value` arrays) must have the **same length**, representing the number of experiments.
- If a particular experiment lacks a value for a field, use `null` as the array element.
- The extra field list should contain only fields not present in the standard schema. For standard fields, use the original field name directly (do not duplicate in `extra`).
"""

    def _parse_response(self, response: str) -> Any:
        """使用更鲁棒的方法解析 JSON，支持尝试修复截断的 JSON。"""
        try:
            return self.client.extract_json_from_response(response)
        except Exception as e:
            logger.debug(f"Normal JSON parsing failed: {e}")

        # 尝试修复截断
        fixed = self._try_fix_truncated_json(response)
        if fixed is not None:
            logger.info("Successfully fixed truncated JSON")
            return fixed
        return None

    def _try_fix_truncated_json(self, response: str) -> Optional[Any]:
        """尝试修复被截断的 JSON 字符串（例如补全缺失的括号）。"""
        response = response.strip()
        open_brackets = response.count('[')
        close_brackets = response.count(']')
        open_braces = response.count('{')
        close_braces = response.count('}')

        missing_close = (open_brackets - close_brackets) + (open_braces - close_braces)
        if 0 < missing_close < 20:
            fixed_response = response + (']' * (open_brackets - close_brackets)) + ('}' * (open_braces - close_braces))
            try:
                return self.client.extract_json_from_response(fixed_response)
            except Exception:
                pass
        return None

    def _convert_columnar_to_records(self, columnar_data: Any) -> List[Dict[str, Any]]:
        """
        将列式格式转换为列表格式（每个实验一个字典）。
        输入: columnar_data 应为列表，第一个元素是包含标准字段数组和 extra 数组的对象。
        输出: 列表，每个元素是单个实验的字典，包含标准字段和 additional_items。
        """
        if not isinstance(columnar_data, list) or len(columnar_data) == 0:
            logger.warning("Columnar data is not a non-empty list")
            return []

        data_obj = columnar_data[0]
        if not isinstance(data_obj, dict):
            logger.warning("Columnar data object is not a dict")
            return []

        # 提取标准字段数组
        standard_fields = {}
        extra_fields = data_obj.get("extra", [])
        num_experiments = 0

        # 先计算实验数量（从任一数组长度获取）
        for key, value in data_obj.items():
            if key == "extra":
                continue
            if isinstance(value, list):
                num_experiments = len(value)
                standard_fields[key] = value
                break

        if num_experiments == 0:
            # 如果没有标准字段，尝试从 extra 中获取长度
            if extra_fields and isinstance(extra_fields, list) and len(extra_fields) > 0:
                first_extra = extra_fields[0]
                if isinstance(first_extra, dict) and "value" in first_extra and isinstance(first_extra["value"], list):
                    num_experiments = len(first_extra["value"])
            else:
                logger.warning("No arrays found to determine number of experiments")
                return []

        # 补充其他标准字段（如果之前未收集全）
        for key, value in data_obj.items():
            if key == "extra":
                continue
            if key not in standard_fields and isinstance(value, list):
                standard_fields[key] = value
            elif not isinstance(value, list):
                logger.warning(f"Field {key} is not a list, skipping")

        # 构建 records
        records = []
        for i in range(num_experiments):
            record = {}
            # 添加标准字段
            for field_name, values in standard_fields.items():
                if i < len(values):
                    record[field_name] = values[i]
                else:
                    record[field_name] = None

            # 添加额外字段（转换为 additional_items 格式）
            additional_items = []
            for extra in extra_fields:
                if not isinstance(extra, dict):
                    continue
                name = extra.get("name")
                define = extra.get("define")
                dtype = extra.get("type", "string")
                values = extra.get("value", [])
                if i < len(values):
                    value = values[i]
                else:
                    value = None
                # 只添加非空值（可选，根据需求）
                if value is not None or name:  # 保留字段结构
                    additional_items.append({
                        "item_name": name,
                        "define": define,
                        "value": value,
                        "suggested_data_type": dtype
                    })
            if additional_items:
                record["additional_items"] = additional_items

            records.append(record)

        return records

    def _normalize_record(self, rec: Dict) -> Dict:
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
        return rec
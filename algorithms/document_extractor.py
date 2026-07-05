import json
import logging
from typing import List, Dict, Any, Optional
from utils.file_utils import load_json
from utils.llm_client import OpenAIClient

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Extract experiment records from paper text using core fields + additional_items."""
    def __init__(self, llm_client: OpenAIClient, schema_path: str):
        self.client = llm_client
        self.schema = load_json(schema_path)

    def extract(self, paper_text: str, schema=None, table_text: Optional[str] = None) -> List[Dict[str, Any]]:
        if schema is None:
            schema = self.schema
        print(len(schema))
        prompt = self._build_prompt(paper_text, schema ,table_text)
        try:
            response = self.client.call(
                prompt
            )
            print('response :', response)
            pass
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return []

        if not response:
            logger.warning("LLM returned empty response")
            return []

        # 输出响应长度，便于调试
        logger.debug(f"LLM response length: {len(response)} characters")

        parsed = self._parse_response(response)
        if not parsed:
            logger.error("Failed to parse LLM response into JSON")
            # 可选：记录截断的响应片段
            logger.debug(f"Failed response snippet: {response[:500]}...")
            return []

        if not isinstance(parsed, list):
            parsed = [parsed]

        records = []
        for rec in parsed:
            if not isinstance(rec, dict):
                continue
            rec = self._normalize_record(rec)
            records.append(rec)

        logger.info(f"Extracted {len(records)} experiment(s)")
        return records

    def update_schema(self, new_schema) -> str:
        self.schema = new_schema


    def _build_prompt(self, paper_text: str, schema, exp_table: Optional[str]=None) -> str:
        if exp_table is not None:
            return f"""
Hello, as a chemical data extraction engineer, there is currently a paper in the field of chemistry here. Please extract the experimental records from it. I hope that the extracted experimental items can meet my database standards.

## Paper data:
{paper_text}
## Experimental items:
{exp_table}
## Database standards:
{schema}

## Note:
1. Do not omit data
2. If there are additional items or experimental items that do not exist in the database standard items, please also list them as schema format.
## Output format:
```json
Output Format:
[
    {{
        // Extracted standard fields here
        "additional_items": [
            {{
                "item_name": "field_name",
                "define": "field_definition",
                "value": "extracted_value",
                "suggested_data_type": "suggested_data_type"
            }}
            ... if exists
        ]
    }}
    ...
]
```
"""
        else:
            return f"""
Hello, as a chemical data extraction engineer, there is currently a paper in the field of chemistry here. Please extract the experimental records from it. I hope that the extracted experimental items can meet my database standards.

## Paper data:
{paper_text}

## Database standards:
{schema}

## Note:
1. Do not omit data
2. If there are additional items or experimental items that do not exist in the database standard items, please also list them as schema format.
## Output format:
```json
Output Format:
[
    {{
        // Extracted standard fields here
        "additional_items": [
            {{
                "item_name": "field_name",
                "define": "field_definition",
                "value": "extracted_value",
                "suggested_data_type": "suggested_data_type"
            }}
            ... if exists
        ]
    }}
    ...
]
```
"""
    def _parse_response(self, response: str) -> Any:
        """Use LLMClient's extract_json_from_response for robust parsing."""
        return self.client.extract_json_from_response(response)

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
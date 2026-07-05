import json
import logging
from typing import List, Dict, Any, Optional,Tuple
from utils.file_utils import load_json
from utils.llm_client import OpenAIClient
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
logger = logging.getLogger(__name__)

class DocumentPlanningExtractor:
    """Extract experiment records from paper text using core fields + additional_items."""
    def __init__(self, llm_client: OpenAIClient, schema: Dict, chunk_size: int=3):
        self.client = llm_client
        self.schema = schema
        self.chunk_size = chunk_size
        
    def extract(self, paper_text: str, table_text: Optional[str] = None, method: str = "chunk") -> List[Dict[str, Any]]:
        """Main entry point: try direct extraction first, then planned chunk extraction."""
        # Attempt direct extraction
        if method == "full":
            direct_result = self._extract_directly(paper_text, table_text)
            if direct_result:
                logger.info(f"Direct extraction succeeded with {len(direct_result)} records")
            return direct_result
        elif method == "chunk":
            logger.warning("Direct extraction failed or returned empty. Switching to planned chunk extraction.")
            # Fallback: planned chunk extraction
            return self._extract_with_planning(paper_text, table_text)
        return []
    
    def _extract_directly(self, paper_text: str, table_text: Optional[str] = None) -> List[Dict[str, Any]]:

        prompt = self._build_prompt(paper_text,  table_text)
        try:
            response = self.client.call(
                prompt
            )
            print(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return []

        if not response:
            logger.warning("LLM returned empty response")
            return []

        # 输出响应长度，便于调试
        logger.debug(f"LLM response length: {len(response)} characters")

        parsed = self.client.extract_json_from_response(response)
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
    
    # def _extract_with_planning(self, paper_text: str, table_text: Optional[str] = None) -> List[Dict[str, Any]]:
    #     total_records = self._estimate_total_records(paper_text, table_text)
    #     logger.info(f"Estimated total records: {total_records}")
    #     if total_records is None or total_records <= 0:
    #         logger.warning("Could not estimate total records, falling back to heuristic line counting")
    #         return []
    #     ranges = self._generate_record_ranges(total_records, self.chunk_size)
    #     logger.info(f"Chunk ranges: {ranges}")

    #     all_chunk_results = []
    #     for start, end in ranges:
    #         chunk_records = self._extract_chunk(paper_text, table_text, start, end)
        #     if chunk_records:
        #         all_chunk_results.extend(chunk_records)
        #     else:
        #         logger.warning(f"Chunk records {start}-{end} produced no output")

    #     return all_chunk_results
    
    def _extract_with_planning(self, paper_text: str, table_text: Optional[str] = None) -> List[Dict[str, Any]]:
        total_records = self._estimate_total_records(paper_text, table_text)
        logger.info(f"Estimated total records: {total_records}")
        if total_records is None or total_records <= 0:
            logger.warning("Could not estimate total records, falling back to heuristic line counting")
            return []


        logger.info(f"Estimated total records: {total_records}")

        ranges = self._generate_record_ranges(total_records, self.chunk_size)
        logger.info(f"Chunk ranges: {ranges}")
        
        results_dict = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {
                executor.submit(self._extract_chunk, paper_text, table_text, start, end): idx
                for idx, (start, end) in enumerate(ranges)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    chunk_records = future.result()
                    if chunk_records:
                        results_dict[idx] = chunk_records
                    else:
                        logger.warning(f"Chunk records {ranges[idx][0]}-{ranges[idx][1]} produced no output")
                        results_dict[idx] = []
                except Exception as e:
                    logger.error(f"Chunk {ranges[idx]} failed: {e}")
                    results_dict[idx] = []

        merged = []
        for idx in range(len(ranges)):
            if idx in results_dict:
                merged.extend(results_dict[idx])
        return merged
    
    def _extract_chunk(self, paper_text: str, table_text: Optional[str],
        start_row: int, end_row: int) -> List[Dict[str, Any]]:
        """Extract records only for rows in [start_row, end_row] (1‑indexed)."""
        prompt = self._build_chunk_prompt(paper_text, table_text, start_row, end_row)
        try:
            response = self.client.call(prompt)
            if not response:
                return []
            parsed = self.client.extract_json_from_response(response)
            print(parsed)
            if not parsed:
                logger.info(f"Chunk rows {start_row}-{end_row} produced no output")
                return []
            if not isinstance(parsed, list):
                parsed = [parsed]
            records = [self._normalize_record(rec) for rec in parsed if isinstance(rec, dict)]
            logger.info(f"Chunk rows {start_row}-{end_row} extracted {len(records)} records")
            return records

            
        except Exception as e:
            logger.error(f"Chunk extraction error for rows {start_row}-{end_row}: {e}")
            return []
        

    def _build_prompt(self, paper_text: str, exp_table: Optional[str]=None) -> str:
        return f"""
Extract experimental records from the chemistry paper. Only include fields defined in the schema. If a field's value is empty (null, empty string, or missing), omit that field from the output. Minimize whitespace.
Exp_table: {exp_table}
Paper: {paper_text}
Schema: {self.schema}

Output format (JSON array):
```json
[{{"field1": "value1", "field2": "value2"}}, ...]```

- The value corresponding to the extracted field should not be a pronoun as much as possible. If it is a pronoun, please replace it with the content of the reference
"""


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
    def _estimate_total_records(self, paper_text: str, table_text: Optional[str] = None) -> Optional[int]:
        """Ask LLM to return the total number of experimental records (data rows) in the table."""
        prompt = self._build_total_records_prompt(paper_text, table_text)
        
        try:
            response = self.client.call(prompt)
            if not response:
                return 0

        # Parse response: expect a plain integer or JSON like {"total_records": 20}
            parsed = self.client.extract_json_from_response(response)
            if isinstance(parsed, int):
                return parsed
            if isinstance(parsed, dict) and "total_records" in parsed:
                return int(parsed["total_records"])
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], int):
                return parsed[0]
        except Exception as e:
            logger.error(f"Error parsing total records response: {response}")
            return 0
        
        # Try to extract integer from string
        try:
            match = re.search(r'\b(\d+)\b', response)
            if match:
                return int(match.group(1))
            return 0
        except Exception as e:
            logger.error(f"Total records estimation failed: {e}")
            return 0
        
    def _build_total_records_prompt(self, paper_text: str, table_text: Optional[str] = None) -> str:
        exp_table_section = f"Exp_table: {table_text}\n" if table_text else ""
        return f"""
You are given a chemistry paper containing one or more tables of experimental data.
Count how many data rows (experiment records) exist in the main table. Do NOT count the header row.
Return ONLY a JSON object with a key "total_records" and the integer value.
Example: {{"total_records": 20}}
# experiment table:
        {exp_table_section}
        """
    def _generate_record_ranges(self, total_records: int, chunk_size: int) -> List[Tuple[int, int]]:
        ranges = []
        for start in range(1, total_records + 1, chunk_size):
            end = min(start + chunk_size - 1, total_records)
            ranges.append((start, end))
        return ranges
    
        
    def _build_chunk_prompt(self, paper_text: str, table_text: Optional[str],
        start_record: int, end_record: int) -> str:
        schema_str = json.dumps(self.schema, indent=2)
        exp_table_section = f"Exp_table: {table_text}\n" if table_text else ""
        return f"""
Extract experimental records from the chemistry paper, but ONLY for experiment record numbers {start_record} through {end_record} (inclusive).
Record numbers start at 1 for the first data row after the header. Include the column headers to understand context.
Only include fields defined in the schema. If a field's value is empty (null, empty string, or missing), omit that field from the output. Minimize whitespace.
{exp_table_section}
Paper: {paper_text}
Schema: {schema_str}

Output format (JSON array):
```json
[{{"field1": "value1", "field2": "value2"}}, ...]```"""
    @staticmethod
    def _merge_records(chunk_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        merged = []
        for chunk in chunk_results:
            merged.extend(chunk)
        return merged
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import logging

logger = logging.getLogger(__name__)

def build_block_audit_prompt(
    block_text: str,
    fields_to_audit: List[Tuple[str, str]],  # [(field_name, extracted_value), ...]
    schema_hint: str = ""
) -> str:
    """
    构建一个针对某个 block 的多字段审核 prompt。
    fields_to_audit: 该 block 中出现的字段名及提取值列表。
    """
    # 构建字段描述部分
    fields_desc = "\n".join([f"- {field}: {value}" for field, value in fields_to_audit])
    
    prompt = f"""You are an expert in chemical experiment data extraction verification.  
You are given a piece of context (text and optionally an image) from a scientific paper.  
Based **only** on this context, please evaluate the correctness of each extracted field listed below.

### Context Text (from paper block)
{block_text if block_text else "(No text provided)"}

### Image
{"An image is attached. Please use it as primary evidence when conflicting with text." if True else "No image for this block."}

### Extracted Fields to Verify
{fields_desc}

### Schema (optional reference)
{schema_hint if schema_hint else "No schema provided."}

### Instructions
For each field, output:
- `confidence`: a float between 0.0 and 1.0, where:
    - 1.0 = exactly correct, clearly stated in context.
    - 0.8-0.9 = correct but minor paraphrasing/unit differences.
    - 0.5-0.7 = partially correct (e.g., missing a modifier, ambiguous).
    - 0.1-0.4 = likely wrong, but some weak connection.
    - 0.0 = no evidence or clearly contradictory.
- `suggested_value`: if confidence < 0.8, provide the most likely correct value (as string); otherwise null.
- `reason`: brief explanation.

### Output Format (JSON only, no extra text)
{{
    "fields": [
        {{"field": "catalyst_name", "confidence": 1.0, "suggested_value": null, "reason": "Explicitly stated"}},
        {{"field": "temperature_c", "confidence": 0.3, "suggested_value": "120", "reason": "Text says 120°C, extracted 80"}}
    ]
}}

Now output the JSON for the above fields.
"""
    return prompt

def audit_by_block(
    records: List[Dict[str, Any]],
    evidence_list: List[Dict[str, str]],
    blocks: List[Block],
    mllm_client,          # 多模态客户端，需支持 call(prompt, images) 方法
    schema_desc: str = "",
    image_cache_dir: str = None,
) -> List[Dict[str, Any]]:
    """
    按照证据块（block）聚合审核，每个 block 调用一次 MLLM。

    Returns:
        审核结果，格式与逐字段调用一致：
        [
            {
                "_record_id": ...,
                "fields": [
                    {
                        "field": "catalyst_name",
                        "original_value": "...",
                        "confidence": 0.95,
                        "suggested_value": null,
                        "reason": "...",
                        "evidence_block_id": "48",
                        "text_context": "...",
                        "image_used": true
                    },
                    ...
                ]
            },
            ...
        ]
    """
    if not records or not evidence_list:
        return []
    
    # 构建 block_id -> Block 映射
    block_dict = {str(block.id): block for block in blocks}
    
    # 第一步：按 block_id 聚合需要审核的字段
    # 结构: block_id -> list of (record_index, field_name, extracted_value)
    block_to_fields = defaultdict(list)
    # 同时记录每个字段所属的记录索引和原始值，以便最后填充结果
    field_metadata = {}  # key: (record_idx, field_name) -> block_id, extracted_value
    
    for rec_idx, (record, evidence) in enumerate(zip(records, evidence_list)):
        for field_name, extracted_value in record.items():
            if field_name.startswith("_"):
                continue
            block_id = evidence.get(field_name)
            if not block_id:
                continue  # 无证据的字段不审核（或单独处理为低分）
            block_id_str = str(block_id)
            block_to_fields[block_id_str].append((rec_idx, field_name, extracted_value))
            field_metadata[(rec_idx, field_name)] = {
                "block_id": block_id_str,
                "value": extracted_value
            }
    
    # 存储每个字段的审核结果
    audit_results_map = {}  # (rec_idx, field_name) -> result dict
    
    # 第二步：对每个有字段的 block 调用 MLLM
    for block_id, field_list in block_to_fields.items():
        block = block_dict.get(block_id)
        if not block:
            logger.warning(f"Block {block_id} not found, skipping {len(field_list)} fields")
            # 为这些字段生成默认低分结果
            for rec_idx, field_name, value in field_list:
                audit_results_map[(rec_idx, field_name)] = {
                    "field": field_name,
                    "original_value": value,
                    "confidence": 0.0,
                    "suggested_value": None,
                    "reason": f"Evidence block {block_id} not found",
                    "evidence_block_id": block_id,
                    "text_context": "",
                    "image_used": False
                }
            continue
        
        # 准备该 block 的文本和图像
        text_context = block.content if block.content else ""
        image = None
        if block.type == "image" and block.image_path:
            full_path = block.image_path
            if image_cache_dir and not os.path.isabs(full_path):
                full_path = os.path.join(image_cache_dir, full_path)
            image = load_image_from_path(full_path)  # 复用之前的加载函数
        has_image = image is not None
        
        # 构建 prompt：传入该 block 下所有 (field, value)
        fields_for_prompt = [(field_name, value) for (_, field_name, value) in field_list]
        prompt = build_block_audit_prompt(
            block_text=text_context,
            fields_to_audit=fields_for_prompt,
            schema_hint=schema_desc
        )
        
        # 调用 MLLM
        try:
            images = [image] if image else []
            response = mllm_client.call(prompt, images=images)
            # 解析 JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON in response")
            result_json = json.loads(json_match.group())
            # 期望得到 {"fields": [...]}
            fields_result = result_json.get("fields", [])
            # 建立字段名到审核结果的映射
            result_by_field = {item["field"]: item for item in fields_result if "field" in item}
            
            # 将审核结果填入 audit_results_map
            for rec_idx, field_name, value in field_list:
                if field_name in result_by_field:
                    res = result_by_field[field_name]
                    audit_results_map[(rec_idx, field_name)] = {
                        "field": field_name,
                        "original_value": value,
                        "confidence": res.get("confidence", 0.0),
                        "suggested_value": res.get("suggested_value"),
                        "reason": res.get("reason", ""),
                        "evidence_block_id": block_id,
                        "text_context": text_context[:500],
                        "image_used": has_image
                    }
                else:
                    # 模型没有返回该字段，给默认低分
                    audit_results_map[(rec_idx, field_name)] = {
                        "field": field_name,
                        "original_value": value,
                        "confidence": 0.0,
                        "suggested_value": None,
                        "reason": "Field not evaluated by MLLM",
                        "evidence_block_id": block_id,
                        "text_context": text_context[:500],
                        "image_used": has_image
                    }
        except Exception as e:
            logger.error(f"Error auditing block {block_id}: {e}")
            # 该 block 下所有字段都标记为低分
            for rec_idx, field_name, value in field_list:
                audit_results_map[(rec_idx, field_name)] = {
                    "field": field_name,
                    "original_value": value,
                    "confidence": 0.0,
                    "suggested_value": None,
                    "reason": f"MLLM call failed: {str(e)}",
                    "evidence_block_id": block_id,
                    "text_context": text_context[:500],
                    "image_used": has_image
                }
    
    # 第三步：将审核结果组装成与 records 相同顺序的输出
    final_results = []
    for rec_idx, record in enumerate(records):
        rec_result = {
            "_record_id": record.get("_record_id", f"record_{rec_idx}"),
            "fields": []
        }
        for field_name, value in record.items():
            if field_name.startswith("_"):
                continue
            key = (rec_idx, field_name)
            if key in audit_results_map:
                rec_result["fields"].append(audit_results_map[key])
            else:
                # 没有证据的字段：给出低分
                rec_result["fields"].append({
                    "field": field_name,
                    "original_value": value,
                    "confidence": 0.0,
                    "suggested_value": None,
                    "reason": "No evidence block provided",
                    "evidence_block_id": None,
                    "text_context": "",
                    "image_used": False
                })
        final_results.append(rec_result)
    
    return final_results
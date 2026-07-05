
from collections import defaultdict
from typing import List, Dict, Any
import json
from .data_structure import Block
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def group_fields_by_block(
    records: List[Dict[str, Any]],
    evidence_list: List[Dict[str, str]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    将每条记录中的字段按照证据 block_id 分组。
    
    返回格式：
    {
        "28": [
            {"catalyst_ligand_type": "value1", "catalyst_name": "value2", ...},  # 来自记录0的字段
            {"catalyst_ligand_type": "value3", "catalyst_name": "value4", ...},  # 来自记录1的字段
            ...
        ],
        "14": [...]
    }
    注意：每个内层字典只包含映射到该 block_id 的字段（可能不包含记录的所有字段）。
    """
    block_to_records = defaultdict(list)
    
    for rec_idx, (record, evidence) in enumerate(zip(records, evidence_list)):
        # 收集该记录中映射到某个 block 的字段
        block_fields = defaultdict(dict)
        for field_name, value in record.items():
            if field_name.startswith("_"):
                continue
            block_id = evidence.get(field_name)
            if not block_id:
                continue
            block_fields[block_id][field_name] = value
        
        # 将该记录的每个 block 子集加入到结果中
        for block_id, fields_dict in block_fields.items():
            # 可选：添加一个 _record_index 便于追溯
            fields_dict["_record_index"] = rec_idx
            block_to_records[block_id].append(fields_dict)
    
    return dict(block_to_records)

def convert_records_to_natural_language(records):
    display_records = []
    for rf in records:
        display_rf = {k: v for k, v in rf.items() if not k.startswith("_")}
        display_records.append(display_rf)
    
    records_json = json.dumps(display_records, indent=2, ensure_ascii=False)
    return records_json
def build_natural_prompt_for_block(
    block_text: str,
    records: List[Dict[str, Any]],
    audit_records: List[Dict[str, Any]],   # 每个元素是一个字典，包含多个字段及其值
    schema_str: Dict
) -> str:
    """
    生成自然风格的审核 prompt。
    records_fields 示例：
    [
        {"catalyst_ligand_type": "X", "catalyst_name": "Y", "_record_index": 0},
        {"catalyst_ligand_type": "A", "catalyst_name": "B", "_record_index": 1}
    ]
    """
    # 去掉内部 _record_index 后再显示
    # display_records = []
    # for rf in records:
    #     display_rf = {k: v for k, v in rf.items() if not k.startswith("_")}
    #     display_records.append(display_rf)
    
    # records_json = json.dumps(display_records, indent=2, ensure_ascii=False)
    records_json = convert_records_to_natural_language(records)
    audit_records_json = convert_records_to_natural_language(audit_records)
    prompt = f"""这是我根据这段图像文本：
{block_text}

提取的实验记录部分（JSON数组，每个元素是一条记录）：
{records_json}

待审核的实验记录部分：
{audit_records_json}
字段名定义（schema）：
{schema_str}


请基于图像，对待审核的每条记录的每个字段进行审核。

要求：
- 对于每条记录，只输出 **错误或需要修正** 的字段。正确的字段不要输出。
- 每个错误字段的输出格式： {{"字段名": ["建议修正的值", 置信度]}}
  - 置信度范围 0.0~1.0，表示你对“原始值错误且建议值正确”的把握程度。
  - 如果字段缺失、格式错误、数值明显不符，给出最可能的正确值；如果不确定，给出 null。
- 输出是一个 JSON 数组，长度与输入 records 相同。每个元素是一个对象（可能为空对象）。
- 如果某条记录所有字段都正确，输出 {{}}。

示例输入 records：
[
  {{"catalyst_name": "Pd/C", "temperature_c": "120"}},
  {{"catalyst_name": "Ni", "temperature_c": "80"}}
]

示例输出：
[
  {{}},
  {{"catalyst_name": ["Pd(PPh3)4", 0.85], "temperature_c": ["100", 0.6]}}
]
（第二条记录两个字段都有错误）

只输出 JSON，不要有任何额外文本。
"""
    return prompt

def audit_by_block_natural(
    records: List[Dict[str, Any]],
    evidence_list: List[Dict[str, str]],
    blocks: List[Block],
    mllm_client,
    schema_desc: Dict = {},
    image_cache_dir: str = None,
) -> List[Dict[str, Any]]:
    """
    按照 block 聚合审核，输出结果与原始记录顺序一致。
    """
    # 1. 按 block 分组
    block_groups = group_fields_by_block(records, evidence_list)
    
    # 2. 建立 block_id -> Block 映射
    block_dict = {str(b.id): b for b in blocks}
    
    # 3. 存储审核结果，key = (record_index, field_name)
    audit_map = {}
    
    # 4. 对每个 block 调用 MLLM

    for block_id, record_fields_list in block_groups.items():
        block = block_dict.get(str(block_id))

        if not block:
            logger.warning(f"Block {block_id} not found, skip {len(record_fields_list)} record groups")
            # 标记为低分
            for rf in record_fields_list:
                rec_idx = rf.get("_record_index")
                for fname, fval in rf.items():
                    if fname.startswith("_"):
                        continue
                    audit_map[(rec_idx, fname)] = {
                        "confidence": 0.0,
                        "suggested_value": None,
                        "reason": f"Block {block_id} not found",
                        "evidence_block_id": block_id
                    }
            continue
        
        # 准备文本和图像
        text_context = block.content if block.content else ""
        image = None


        has_image = image is not None
        #image_url = image_to_base64_data_uri(block.image_path) if has_image else None
        # 构建 prompt
        prompt = build_natural_prompt_for_block(text_context, records, record_fields_list, schema_desc)
        
        try:
            images = [image] if image else []
            response = mllm_client.call_with_image(prompt, block.image_path)

            parsed = mllm_client.extract_json_from_response(response)

                        
            # 解析结果
            for idx, record_result in enumerate(parsed):
                if idx >= len(record_fields_list):
                    print("idx >= len(record_fields_list)", idx, len(record_fields_list))
                    break
                rec_idx = record_fields_list[idx].get("_record_index")
                if rec_idx is None:
                    continue
                
                # 获取该记录在当前 block 下的所有字段（待审核字段）
                fields_in_this_block = {k: v for k, v in record_fields_list[idx].items() if not k.startswith("_")}
                
                # 模型返回的错误字段集合
                returned_fields = set(record_result.keys())
                
                # 1. 处理模型明确返回的错误字段
                for field_name, value in record_result.items():
                    if not isinstance(value, list) or len(value) < 2:
                        continue
                    suggested = value[0]
                    try:
                        confidence = float(value[1])
                    except:
                        confidence = 0.0
                    audit_map[(rec_idx, field_name)] = {
                        "confidence": confidence,
                        "suggested_value": suggested,
                        "reason": "Reviewed by MLLM (error)",
                        "evidence_block_id": block_id
                    }
                
                # 2. 处理模型未返回的正确字段（即该 block 下应有但模型没报错的字段）
                for field_name in fields_in_this_block:
                    if field_name not in returned_fields:
                        audit_map[(rec_idx, field_name)] = {
                            "confidence": 1.0,
                            "suggested_value": None,
                            "reason": "Correct (not flagged by MLLM)",
                            "evidence_block_id": block_id
                        }
        except Exception as e:
            logger.error(f"Error auditing block {block_id}: {e}")
            for rf in record_fields_list:
                rec_idx = rf.get("_record_index")
                for fname, fval in rf.items():
                    if fname.startswith("_"):
                        continue
                    audit_map[(rec_idx, fname)] = {
                        "confidence": 0.0,
                        "suggested_value": None,
                        "reason": f"MLLM call failed: {e}",
                        "evidence_block_id": block_id
                    }

        
    # 5. 组装最终输出（按原始记录顺序）
    final_results = []
    for rec_idx, record in enumerate(records):
        merged = {}
        # 保留 _record_id（如果存在）
        if "_record_id" in record:
            merged["_record_id"] = record["_record_id"]
        
        for field_name, original_value in record.items():
            if field_name.startswith("_"):
                continue
            key = (rec_idx, field_name)
            if key in audit_map:
                info = audit_map[key]
                merged[field_name] = {
                    "original_value": original_value,
                    "suggested_value": info["suggested_value"],
                    "confidence": info["confidence"],
                    "block_id": info["evidence_block_id"],
                    "reason": info["reason"]
                }
            else:
                merged[field_name] = {
                    "original_value": original_value,
                    "suggested_value": None,
                    "confidence": 0.0,
                    "block_id": None,
                    "reason": "No evidence or not reviewed"
                }
        final_results.append(merged)

    return final_results
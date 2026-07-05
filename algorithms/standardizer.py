
import os
import logging
from typing import Dict, Any, List, Optional
from utils.file_utils import save_json, load_json

logger = logging.getLogger(__name__)

class Standardizer:
    """
    基础标准化器：
    - 将 DocumentExtractor 输出的原始记录转换为内部格式。
    - 收集额外字段的元数据（名称、定义、值、类型），用于后续批量合并。
    - 不进行任何字段名匹配，只做标记。
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.raw_records_file = os.path.join(data_dir, "raw_records.json")
        self.standardized_records_file = os.path.join(data_dir, "standardized_records.json")
        self.extra_fields_meta_file = os.path.join(data_dir, "extra_fields_meta.json")
        os.makedirs(data_dir, exist_ok=True)

    def standardize(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        标准化记录列表，并收集额外字段元数据。
        返回标准化后的记录（字段名不变，但额外字段标记为 _extra）。
        """
        standardized = []
        extra_meta = {}  # {原始字段名: {"definition": ..., "suggested_type": ..., "values": [], "record_ids": []}}

        for rec in records:
            std_rec = {}
            for field, value in rec.items():
                if field.startswith("_"):  # 元数据保留
                    std_rec[field] = value
                    continue
                # 检查是否为额外字段（DocumentExtractor 添加的标记）
                if isinstance(value, dict) and value.get("_is_extra"):
                    original_name = field
                    definition = value.get("definition", "")
                    suggested_type = value.get("suggested_type", "string")
                    raw_value = value.get("value", "")
                    # 收集元数据
                    if original_name not in extra_meta:
                        extra_meta[original_name] = {
                            "definition": definition,
                            "suggested_type": suggested_type,
                            "values": [],
                            "record_ids": []
                        }
                    extra_meta[original_name]["values"].append(raw_value)
                    extra_meta[original_name]["record_ids"].append(rec.get("_record_id", "unknown"))
                    # 保留原始字段，但改为简单值（去掉 _is_extra 标记）
                    std_rec[original_name] = raw_value
                else:
                    # 标准字段直接保留
                    std_rec[field] = value
            standardized.append(std_rec)

        # 保存原始记录（DocumentExtractor 输出）
        save_json(records, self.raw_records_file)
        # 保存标准化后的记录（字段名未变，但值简化）
        save_json(standardized, self.standardized_records_file)
        # 保存额外字段元数据
        save_json(extra_meta, self.extra_fields_meta_file)

        logger.info(f"Standardized {len(records)} records, found {len(extra_meta)} unique extra fields.")
        return standardized
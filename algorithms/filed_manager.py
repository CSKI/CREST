import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional
from utils.llm_client import LLMClient, OpenAIClient   # 假设您的文件名为 llm_client.py

# ==========================
# 1. 工具函数：单位提取与追加
# ==========================
def extract_unit(value_str: str) -> str:
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*([a-zA-Z°%]+|g/mol|mol/L|kg/mol|MPa|kPa|bar|psi|min|h|s|ms|μmol|mmol|mol|L|mL|g|kg|mg|μg|°C|°F|K|%|ppm|ppb)', value_str)
    if match:
        return match.group(1).strip()
    return ""

def append_unit_to_definition(definition: str, value: str) -> str:
    unit = extract_unit(value)
    if unit and unit not in definition:
        if definition.endswith('.'):
            return f"{definition} Unit: {unit}."
        else:
            return f"{definition}. Unit: {unit}."
    return definition
# ==========================
# 新增：统计模块
# ==========================
class SchemaStatistics:
    """
    从标准化后的文档列表中计算各种统计指标，用于绘制论文中的四张图。
    """

    def __init__(self, docs: List[Dict], pruning_threshold: float = 0.8, active_threshold: float = 0.2):
        """
        Args:
            docs: 经过 ExtraFieldStandardizer 处理后的文档列表，按论文顺序排列。
            pruning_threshold: 字段空值率高于此阈值时被视为“可修剪”（不活跃）。
            active_threshold: 字段空值率低于此阈值时被视为“活跃核心字段”（高填充率）。
                               注意：active_threshold 应当 <= pruning_threshold，通常 active_threshold 更严格。
        """
        self.docs = docs
        self.pruning_threshold = pruning_threshold
        self.active_threshold = active_threshold
        self._all_fields = None          # 缓存所有字段名
        self._field_null_rates = None    # 缓存最终空值率

    def _extract_value(self, field_value: Any) -> Any:
        """从字段值中提取真正的数据值（处理 _is_extra 字典）"""
        if isinstance(field_value, dict) and field_value.get('_is_extra'):
            return field_value.get('value')
        return field_value

    def _is_missing(self, value: Any) -> bool:
        """判断一个值是否缺失（None、空字符串、空列表等）"""
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        return False

    def _compute_field_null_rates(self, docs: List[Dict]) -> Dict[str, float]:
        """计算每个字段在给定文档列表中的空值率"""
        field_counts = defaultdict(int)      # 字段出现次数（文档中有该字段）
        field_missing = defaultdict(int)     # 字段缺失次数（文档中无该字段或值为空）
        for doc in docs:
            for field, raw_val in doc.items():
                # 跳过内部辅助字段
                if field.startswith('_'):
                    continue
                field_counts[field] += 1
                val = self._extract_value(raw_val)
                if self._is_missing(val):
                    field_missing[field] += 1
            # 对于本文档未出现的字段，也算一次缺失（需要记录所有字段的基数）
            # 先收集所有出现过字段的集合，最后统一补全缺失计数
        # 补全缺失计数：对于任意字段，文档总数减去出现次数就是缺失次数
        total_docs = len(docs)
        all_fields = set(field_counts.keys())
        null_rates = {}
        for field in all_fields:
            present = field_counts[field]
            missing = total_docs - present + field_missing.get(field, 0)   # 注意：field_missing 已经包含了“有字段但值为空”的情况
            # 更准确：缺失 = 文档中没有该字段 + 有字段但值为空
            # 上面 field_counts 统计的是“字段存在”的次数（不论值是否为空），所以缺失1 = total_docs - field_counts[field]
            # 然后再加上 field_missing（有字段但值为空）并不会重复，因为 field_counts 包含了这些有字段的文档。
            # 实际上，空值率 = (文档中字段缺失 + 文档中字段存在但值为空) / total_docs
            # 文档中字段缺失 = total_docs - field_counts[field]
            # 文档中字段存在但值为空 = field_missing[field]
            missing_total = (total_docs - field_counts[field]) + field_missing.get(field, 0)
            null_rates[field] = missing_total / total_docs
        self._all_fields = all_fields
        return null_rates

    def get_null_rate_distribution(self) -> Dict[str, float]:
        """返回所有字段的最终空值率（用于图1）"""
        if self._field_null_rates is None:
            self._field_null_rates = self._compute_field_null_rates(self.docs)
        return self._field_null_rates

    def get_pruning_summary(self) -> Dict[str, Any]:
        """
        返回与剪枝相关的汇总统计（对应图1中的数字标签）
        """
        null_rates = self.get_null_rate_distribution()
        total_fields = len(null_rates)
        if total_fields == 0:
            return {}
        candidates_pruning = sum(1 for r in null_rates.values() if r > self.pruning_threshold)
        candidates_retention = total_fields - candidates_pruning
        # 活跃核心字段：空值率低于 active_threshold
        active_core = sum(1 for r in null_rates.values() if r <= self.active_threshold)
        return {
            "total_fields": total_fields,
            "candidates_for_pruning": candidates_pruning,
            "candidates_for_pruning_percent": candidates_pruning / total_fields * 100,
            "candidates_for_retention": candidates_retention,
            "candidates_for_retention_percent": candidates_retention / total_fields * 100,
            "active_core_fields": active_core,
            "active_core_fields_percent": active_core / total_fields * 100,
            "pruning_threshold": self.pruning_threshold,
            "active_threshold": self.active_threshold,
        }

    def compute_dynamics(self) -> Dict[str, List]:
        """
        按论文顺序模拟动态过程，返回：
        - paper_indices: 论文序号 (1-based)
        - cumulative_discovered: 累计发现的不同字段数
        - net_active: 当前活跃字段数（空值率 <= pruning_threshold）
        - pruning_events: 列表，每个元素为 (paper_index, field_name)
        """
        cumulative_fields = set()
        active_fields = set()
        pruning_events = []
        cumulative_counts = []
        active_counts = []
        paper_indices = []

        # 记录每个字段最后一次变为不活跃时的论文索引（避免重复记录同一字段多次）
        pruned_recorded = set()

        for i, doc in enumerate(self.docs, start=1):
            # 当前文档包含的字段名（排除内部辅助字段）
            current_fields = {k for k in doc.keys() if not k.startswith('_')}
            cumulative_fields.update(current_fields)

            # 重新计算所有已发现字段在当前累积文档中的空值率
            cum_docs = self.docs[:i]
            null_rates = self._compute_field_null_rates(cum_docs)

            # 更新活跃字段集
            new_active = {f for f in cumulative_fields if null_rates.get(f, 1.0) <= self.pruning_threshold}
            # 检查哪些字段从活跃变为不活跃（剪枝事件）
            for f in active_fields - new_active:
                if f not in pruned_recorded:
                    pruning_events.append((i, f))
                    pruned_recorded.add(f)
            active_fields = new_active

            cumulative_counts.append(len(cumulative_fields))
            active_counts.append(len(active_fields))
            paper_indices.append(i)

        return {
            "paper_indices": paper_indices,
            "cumulative_discovered": cumulative_counts,
            "net_active": active_counts,
            "pruning_events": pruning_events,
        }

    def compute_rank_displacement(self) -> List[Dict]:
        """
        计算每个字段的排名位移：
        - 首次出现时的频率排名（基于当时已处理文档中该字段的出现次数）
        - 最终频率排名（基于全部文档）
        - 位移 = 最终排名 - 首次排名（正值表示排名下降，负值表示上升）
        返回列表，每个元素包含字段名、首次排名、最终排名、位移。
        """
        total_docs = len(self.docs)
        # 最终频率：字段在所有文档中的非缺失出现次数
        final_freq = {}
        for field in self.get_null_rate_distribution().keys():
            cnt = 0
            for doc in self.docs:
                if field in doc:
                    val = self._extract_value(doc[field])
                    if not self._is_missing(val):
                        cnt += 1
            final_freq[field] = cnt
        # 按频率降序排序得到最终排名（1为最高频）
        final_rank = {field: idx+1 for idx, (field, _) in enumerate(sorted(final_freq.items(), key=lambda x: x[1], reverse=True))}

        # 首次出现时的频率排名
        first_rank = {}
        field_first_doc = {}   # 记录首次出现的文档索引
        for i, doc in enumerate(self.docs, start=1):
            current_fields = [f for f in doc.keys() if not f.startswith('_')]
            for f in current_fields:
                if f not in field_first_doc:
                    field_first_doc[f] = i
                    # 计算此时（前i篇文档）中该字段的频率
                    freq_at_first = 0
                    for prev_doc in self.docs[:i]:
                        if f in prev_doc:
                            val = self._extract_value(prev_doc[f])
                            if not self._is_missing(val):
                                freq_at_first += 1
                    # 此时所有已出现字段的频率分布
                    all_fields_up_to_i = set()
                    for prev_doc in self.docs[:i]:
                        all_fields_up_to_i.update([k for k in prev_doc.keys() if not k.startswith('_')])
                    freq_dict = {}
                    for ff in all_fields_up_to_i:
                        cnt = 0
                        for prev_doc in self.docs[:i]:
                            if ff in prev_doc:
                                val = self._extract_value(prev_doc[ff])
                                if not self._is_missing(val):
                                    cnt += 1
                        freq_dict[ff] = cnt
                    sorted_fields = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                    rank = {field: idx+1 for idx, (field, _) in enumerate(sorted_fields)}.get(f, len(sorted_fields)+1)
                    first_rank[f] = rank

        # 构建结果
        results = []
        for field in final_rank:
            if field in first_rank:
                disp = final_rank[field] - first_rank[field]
                results.append({
                    "field": field,
                    "rank_at_discovery": first_rank[field],
                    "rank_at_mature": final_rank[field],
                    "displacement": disp,
                    "first_appeared_at_paper": field_first_doc.get(field, 0)
                })
        return results

    def get_schema_state_at_n(self, n: int = 100) -> Dict[str, Any]:
        """
        返回前 n 篇论文后的自适应模式状态：
        - schema_fields (active core): 活跃核心字段数量（空值率 <= active_threshold）
        - non_schema_discovered_fields: 所有发现字段中非活跃的数量（空值率 > pruning_threshold）
        注意：这里“non-schema discovered fields”指那些被发现了但从未成为核心的字段。
        """
        if n > len(self.docs):
            n = len(self.docs)
        docs_subset = self.docs[:n]
        null_rates = self._compute_field_null_rates(docs_subset)
        total_discovered = len(null_rates)
        active_core = sum(1 for r in null_rates.values() if r <= self.active_threshold)
        # 非活跃发现字段 = 总发现字段 - 活跃核心字段
        non_active = total_discovered - active_core
        return {
            "papers_processed": n,
            "total_discovered_fields": total_discovered,
            "active_core_fields": active_core,
            "active_core_percent": active_core / total_discovered * 100 if total_discovered else 0,
            "non_schema_discovered_fields": non_active,
            "non_schema_percent": non_active / total_discovered * 100 if total_discovered else 0,
        }
# ==========================
# 2. 缓存管理器（人工确认的字段映射）
# ==========================
class FieldMappingCache:
    def __init__(self, cache_file_path: Optional[str] = None):
        self.mapping = {}          # {原始字段名: 标准字段名}
        self.cache_file_path = cache_file_path
        if cache_file_path:
            self.load(cache_file_path)

    def load(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
            print(f"成功加载缓存映射，共 {len(self.mapping)} 条规则")
        except FileNotFoundError:
            print(f"缓存文件 {file_path} 不存在，将创建新缓存")
            self.mapping = {}
        except Exception as e:
            print(f"加载缓存失败: {e}")
            self.mapping = {}

    def save(self, file_path: Optional[str] = None):
        path = file_path or self.cache_file_path
        if not path:
            raise ValueError("未指定缓存文件路径")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, indent=2, ensure_ascii=False)
        print(f"缓存已保存至 {path}")

    def get(self, original_field: str) -> Optional[str]:
        return self.mapping.get(original_field)

    def add(self, original_field: str, standard_field: str):
        self.mapping[original_field] = standard_field

    def record_unmapped(self, unmapped_fields: List[str], output_file: str = "unmapped_fields.json"):
        existing = set()
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = set(json.load(f))
        except:
            pass
        new_fields = set(unmapped_fields) - existing
        if new_fields:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(list(existing | new_fields), f, indent=2, ensure_ascii=False)
            print(f"记录了 {len(new_fields)} 个未匹配字段到 {output_file}")

# ==========================
# 3. 智能体核心类（缓存优先 + LLM 回退）
# ==========================
class ExtraFieldStandardizer:
    def __init__(self,
                 cache: FieldMappingCache,
                 llm_client: Optional[LLMClient] = None,
                 standard_schema: Dict = None,
                 fallback_to_llm: bool = False,
                 batch_size: int = 10):
        """
        cache: 字段映射缓存
        llm_client: 您的 LLMClient 实例（如果 fallback_to_llm=True 则必须提供）
        standard_schema: 标准字段定义字典，格式 {字段名: {"Description & StandardDefinition": "...", ...}}
        fallback_to_llm: 对于缓存中不存在的字段，是否调用 LLM 进行智能合并
        batch_size: 批次大小（仅在 fallback_to_llm=True 时用于分批调用 LLM）
        """
        self.cache = cache
        self.llm_client = llm_client
        self.standard_schema = standard_schema or {}
        self.fallback_to_llm = fallback_to_llm
        self.batch_size = batch_size
        self.unmapped_fields_in_batch = set()

    def _is_extra_field(self, value: Any) -> bool:
        return isinstance(value, dict) and value.get("_is_extra") is True

    # ---------- 基于缓存的标准化 ----------
    def _apply_cache_mapping(self, docs: List[Dict]) -> List[Dict]:
        for doc in docs:
            extra_items = [(k, v) for k, v in doc.items() if self._is_extra_field(v)]
            for orig_name, field_value in extra_items:
                target = self.cache.get(orig_name)
                if target:
                    del doc[orig_name]
                    # 合并到目标字段（如果已经存在）
                    if target in doc and self._is_extra_field(doc[target]):
                        existing = doc[target]
                        existing["evidence"] += f", {field_value.get('evidence', '')}"
                        existing.setdefault("_original_names", []).append(orig_name)
                    else:
                        new_field = field_value.copy()
                        new_field["_original_names"] = [orig_name]
                        if target in self.standard_schema:
                            std_def = self.standard_schema[target]["Description & StandardDefinition"]
                            value_str = new_field.get("value", "")
                            new_field["definition"] = append_unit_to_definition(std_def, value_str)
                        new_field["_is_extra"] = True
                        doc[target] = new_field
                else:
                    self.unmapped_fields_in_batch.add(orig_name)
                    field_value["_unmapped"] = True
        return docs

    # ---------- LLM 回退：对未匹配字段进行智能合并 ----------
    def _call_llm_for_unmapped(self, docs: List[Dict]) -> Dict[str, str]:
        """
        收集所有未匹配字段的信息，调用 LLM 获取映射。
        返回 {原始字段名: 标准字段名}
        """
        # 收集未匹配字段的元信息（去重）
        unmapped_info = []
        seen = set()
        for doc in docs:
            for k, v in doc.items():
                if self._is_extra_field(v) and v.get("_unmapped", False) and k not in seen:
                    seen.add(k)
                    unmapped_info.append({
                        "field_name": k,
                        "definition": v.get("definition", ""),
                        "example_value": v.get("value", "")
                    })
        if not unmapped_info:
            return {}

        # 构建标准字段列表描述
        std_desc = "\n".join([
            f"- {name}: {info.get('Description & StandardDefinition', '')}"
            for name, info in self.standard_schema.items()
        ]) if self.standard_schema else "（无预定义标准字段）"

        extra_desc = "\n".join([
            f"- 字段名: {info['field_name']}, 定义: {info['definition']}, 示例值: {info['example_value']}"
            for info in unmapped_info
        ])

        prompt = f"""
你是一个化学数据标准化专家。下面有一组标准字段（来自化学数据库 schema）和一批从文档中提取的额外字段（带 _is_extra 标记）。

标准字段列表（字段名: 定义）：
{std_desc}

额外字段列表（需要被映射或保留）：
{extra_desc}

任务：
1. 对于每个额外字段，判断它是否可以映射到某个标准字段（语义相同或非常相似）。如果可以，输出时 "summery_name" 使用标准字段名。
2. 如果不能映射到任何标准字段，则你可以创建一个新的标准名（使用英文小写加下划线，例如 "catalyst_amount"），并提供合理的定义。
3. 如果一个标准字段对应多个额外字段（同义词），请将它们归入同一组，relation_name 列出所有原始额外字段名。
4. 如果某个额外字段是唯一且无法归类的，也单独成组，summery_name 可以沿用原始字段名（但建议规范化）。

输出格式为 JSON 数组，每个元素：
{{"summery_name": "标准字段名或新字段名", "sumery_define": "该属性的定义（若映射到标准字段则使用标准定义，否则根据上下文给出）", "relation_name": ["原始额外字段名1", "原始额外字段名2", ...]}}

注意：
- sumery_define 应该包含必要的单位信息（如果示例值中有单位，请在定义中注明单位）。
- 只输出 JSON 数组，不要有其他解释文字。

请开始分析。
"""
        response = self.llm_client.call(prompt)
        if not response:
            return {}
        parsed = self.llm_client.extract_json_from_response(response)
        if not isinstance(parsed, list):
            print(f"LLM 返回的不是数组: {parsed}")
            return {}

        # 构建映射
        mapping = {}
        for group in parsed:
            target = group.get("summery_name")
            if not target:
                continue
            for orig in group.get("relation_name", []):
                mapping[orig] = target
                # 如果是新字段，记录其定义（可选，后续可以写入缓存）
                if target not in self.standard_schema:
                    # 可以存入一个临时字典，供后续使用
                    pass
        return mapping

    def _apply_llm_mapping(self, docs: List[Dict], llm_mapping: Dict[str, str]) -> List[Dict]:
        """将 LLM 返回的映射应用到文档中，并更新缓存"""
        for doc in docs:
            extra_items = [(k, v) for k, v in doc.items() if self._is_extra_field(v) and k in llm_mapping]
            for orig_name, field_value in extra_items:
                target = llm_mapping[orig_name]
                del doc[orig_name]
                # 合并逻辑同缓存
                if target in doc and self._is_extra_field(doc[target]):
                    existing = doc[target]
                    existing["evidence"] += f", {field_value.get('evidence', '')}"
                    existing.setdefault("_original_names", []).append(orig_name)
                else:
                    new_field = field_value.copy()
                    new_field["_original_names"] = [orig_name]
                    if target in self.standard_schema:
                        std_def = self.standard_schema[target]["Description & StandardDefinition"]
                        value_str = new_field.get("value", "")
                        new_field["definition"] = append_unit_to_definition(std_def, value_str)
                    new_field["_is_extra"] = True
                    doc[target] = new_field
                # 将映射添加到缓存中，供后续批次使用
                self.cache.add(orig_name, target)
        return docs

    # ---------- 统计与批次处理 ----------
    def _compute_batch_stats(self, docs: List[Dict]) -> Dict[str, Dict]:
        freq = defaultdict(int)
        def_map = {}
        type_map = {}
        for doc in docs:
            for key, value in doc.items():
                if self._is_extra_field(value):
                    freq[key] += 1
                    if key not in def_map:
                        def_map[key] = value.get("definition", "")
                        type_map[key] = value.get("suggested_type", "")
        summary = {}
        for field in freq:
            summary[field] = {
                "freq": freq[field],
                "definition": def_map.get(field, ""),
                "suggested_type": type_map.get(field, "")
            }
        return summary

    def _attach_summary(self, docs: List[Dict], summary: Dict[str, Dict]) -> List[Dict]:
        for doc in docs:
            doc["_extra_field_summary"] = summary
        return docs

    def process_batch(self, docs: List[Dict]) -> List[Dict]:
        """处理一个批次：缓存标准化 -> 可选 LLM 回退 -> 统计 -> 回写"""
        if not docs:
            return docs
        self.unmapped_fields_in_batch.clear()
        # 1. 缓存标准化
        docs = self._apply_cache_mapping(docs)
        # 2. 如果启用 LLM 回退且有未匹配字段，调用 LLM 并应用映射
        if self.fallback_to_llm and self.unmapped_fields_in_batch and self.llm_client:
            llm_mapping = self._call_llm_for_unmapped(docs)
            if llm_mapping:
                docs = self._apply_llm_mapping(docs, llm_mapping)
                # 保存更新后的缓存
                self.cache.save()
        # 3. 记录仍然未匹配的字段（可能 LLM 也未能覆盖）
        if self.unmapped_fields_in_batch:
            self.cache.record_unmapped(list(self.unmapped_fields_in_batch))
        # 4. 统计频次
        summary = self._compute_batch_stats(docs)
        docs = self._attach_summary(docs, summary)
        return docs

    def process_all(self, all_docs: List[Dict]) -> List[Dict]:
        updated = []
        for i in range(0, len(all_docs), self.batch_size):
            batch = all_docs[i:i+self.batch_size]
            updated_batch = self.process_batch(batch)
            updated.extend(updated_batch)
        return updated

    @staticmethod
    def get_all_extra_fields(docs: List[Dict]) -> Dict[str, int]:
        freq = defaultdict(int)
        for doc in docs:
            for key, value in doc.items():
                if isinstance(value, dict) and value.get("_is_extra"):
                    freq[key] += 1
        return dict(freq)

# ==========================
# 4. 使用示例
# ==========================
if __name__ == "__main__":
    # 假设您已有 LLM 客户端配置
    llm_client = OpenAIClient(
        model_name="gpt-4o",
        base_url="https://api.openai.com/v1",  # 替换为您的 endpoint
        api_key="your-api-key"
    )

    # 加载人工确认的缓存映射
    cache = FieldMappingCache("field_cache.json")

    # 加载标准字段 schema（您提供的 JSON）
    with open("standard_schema.json", "r") as f:
        standard_schema = json.load(f)

    # 创建标准化智能体，启用 LLM 回退
    standardizer = ExtraFieldStandardizer(
        cache=cache,
        llm_client=llm_client,
        standard_schema=standard_schema,
        fallback_to_llm=True,
        batch_size=5
    )

    # 处理一批文档
    sample_docs = [...]  # 您的文档列表
    processed_docs = standardizer.process_all(sample_docs)

    # 查看结果
    print(processed_docs)
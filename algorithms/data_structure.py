from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import threading

@dataclass
class Block:
    """MinerU 输出的块结构（简化）"""
    id: int
    type: str          # "text", "table", "image"
    content: str       # 文本内容或表格的 markdown
    page: int
    # 表格专用
    table_title: str = ""
    table_body: str = ""
    table_footnotes: List[str] = field(default_factory=list)
    # 图像专用
    image_path: str = ""
    image_caption: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,  # 限制长度
            "page": self.page,
            "table_title": self.table_title,
            "table_body": self.table_body if self.table_body else "",
            "table_footnotes": self.table_footnotes,
            "image_caption": self.image_caption,
            "image_path": self.image_path
        }
        
class TokenTracker:
    """线程安全的 Token 统计器"""
    def __init__(self):
        self._use_age = []
        self._lock = threading.Lock()

    def add_usage(self, useage):
        with self._lock:
            self._use_age.append(useage)
    def get_useage(self):
        with self._lock:
            return self._use_age
         
def dicts_to_blocks(blocks_dicts: List[Dict]) -> List[Block]:
    blocks = []
    for d in blocks_dicts:
        
        table_data = d.get("table_data", {})
        block = Block(
            id=d.get("block_id", 0),
            type=d.get("type", "text"),
            content=d.get("content", ""),
            page=d.get("page", 0),
            table_title=table_data.get("table_caption", ""),
            table_body=table_data.get("table_body", ""),
            table_footnotes=table_data.get("table_footnote", "").split('\n') if table_data.get("table_footnote") else [],
            image_path=d.get("image_path", ""),
            image_caption=d.get("content", "") if d.get("type") == "image" else ""
        )
        blocks.append(block)
    return blocks

def build_full_table_context(exp_table: Block, related_blocks: List[Block]):
    context_parts = []
    table_content = ""
    for blk in related_blocks:
        if blk.id == exp_table.id:
            continue
    
        elif blk.type == "image":
            context_parts.append(f"[Block {blk.id} - IMAGE caption]\n{blk.image_caption}\n")
        else:
            context_parts.append(f"[Block {blk.id} - TEXT on page {blk.page}]\n{blk.content}\n")
    table_content = f"[Block {exp_table.id} - TABLE START]\nTable title: {exp_table.table_title}\nTable body:\n{exp_table.table_body}\nTable footnotes: {exp_table.table_footnotes}\n[TABLE END]"
    return "\n".join(context_parts), table_content

def build_full_context(related_blocks: List[Block]):
    context_parts = []
    for blk in related_blocks:
        if blk.type == "table":
            context_parts.append(f"[Block-id {blk.id} - TABLE START]\nTable title: {blk.table_title}\nTable body:\n{blk.table_body}\nTable footnotes: {blk.table_footnotes}\n[TABLE END]")
        elif blk.type == "image":
            context_parts.append(f"[Block-id {blk.id} - IMAGE caption]\n{blk.image_caption}\n")
        else:
            context_parts.append(f"[Block-id {blk.id} - TEXT on page {blk.page}]\n{blk.content}\n")
   
    return "\n".join(context_parts)


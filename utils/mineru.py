import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import fitz
import numpy as np
from .image_tools import pdf_page_to_image, crop_cv_img, convert_bbox_to_high_dpi
# from .layout_process import mk_blocks_to_markdown, merge_bbox


class MinerUParser:
    """
    使用 MinerU 输出的 JSON 和原始 PDF，提取文本、表格和图像，
    并保存表格/图像的裁剪图片。
    """

    def __init__(self, pdf_path: str, json_path: str, cache_dir: str = "./images/", dpi: int = 300):
        self.pdf_path = pdf_path
        self.json_path = json_path
        self.cache_dir = cache_dir
        self.dpi = dpi
        self.doc: Optional[fitz.Document] = None
        self._load_pdf()
    
    def _load_pdf(self) -> None:
        self.doc = fitz.open(self.pdf_path)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_blocks(self) -> List[Dict[str, Any]]:
        if self.doc is None:
            self._load_pdf()
        print(self.json_path)
        with open(self.json_path, 'r', encoding='utf-8') as f:
            layout = json.load(f)
        
        pdf_info = layout.get('pdf_info', [])
        all_blocks = []
        block_id = 0

        for idx, page_info in enumerate(pdf_info):
            page_idx = page_info.get('page_idx', 0)
            para_blocks = page_info.get('preproc_blocks', [])
            
            # 渲染当前页为高DPI图片（numpy数组，BGR格式）
            page_image = pdf_page_to_image(self.doc, page_idx, self.dpi)
            
            # 转换为标准块结构（已包含 children 和合并后的基础 bbox）
            page_markdown = mk_blocks_to_markdown(para_blocks, True, '', self.dpi)
            
            for block_data in page_markdown:
                block_type = block_data.get('type')
                block_content = block_data.get('content', '')
                bbox = block_data.get('bbox', [0, 0, 0, 0])   # 初始 bbox（可能是部分区域）
               
                
                block = {
                    "type": block_type,
                    "content": block_content,
                    "bbox": bbox,
                    "page": page_idx,
                    "convert_type": block_type,
                    "block_id": block_id,
                }
                
                # ========== 处理表格：合并 children 的 bbox，裁剪并保存图片 ==========
                if block_type == "table":
                    children = block_data.get('children', [])
                    if children:
                        # 合并所有子块的 bbox（表格标题、表体、注释等）
                        merged_bbox = [float('inf'), float('inf'), -float('inf'), -float('inf')]
                        table_data = {}
                        for child in children:
                            child_type = child.get('type')
                            child_content = child.get('content', '')
                            table_data[child_type] = child_content
                            child_bbox = child.get('bbox', [0, 0, 0, 0])
                            if len(child_bbox) == 4:
                                merged_bbox[0] = min(merged_bbox[0], child_bbox[0])
                                merged_bbox[1] = min(merged_bbox[1], child_bbox[1])
                                merged_bbox[2] = max(merged_bbox[2], child_bbox[2])
                                merged_bbox[3] = max(merged_bbox[3], child_bbox[3])
                        final_bbox = merged_bbox
                        block["table_data"] = table_data
                    else:
                        final_bbox = bbox
                    
                    # 裁剪表格图片
                    final_bbox = convert_bbox_to_high_dpi(final_bbox, source_dpi=72, target_dpi=self.dpi)
                    cropped_table = crop_cv_img(page_image, final_bbox)
                    # 保存图片
                    safe_bbox = '_'.join(str(int(v)) for v in final_bbox)
                    img_name = f"table_p{page_idx}_b{block_id}_{safe_bbox}.png"
                    img_path = os.path.join(self.cache_dir, img_name)
                    cv2.imwrite(img_path, cropped_table)
                    block["image_path"] = img_path
                    # 更新 bbox 为合并后的完整框
                    block["bbox"] = final_bbox
                
                # ========== 处理图像：直接裁剪并保存 ==========
                elif block_type == "image":
                    bbox = convert_bbox_to_high_dpi(bbox, source_dpi=72, target_dpi=self.dpi)
                    cropped_img = crop_cv_img(page_image, bbox)
                    safe_bbox = '_'.join(str(int(v)) for v in bbox)
                    img_name = f"image_p{page_idx}_b{block_id}_{safe_bbox}.png"
                    img_path = os.path.join(self.cache_dir, img_name)
                    cv2.imwrite(img_path, cropped_img)
                    block["image_path"] = img_path
                else:
                    # ========== 处理表格：裁剪并保存 ==========
                    bbox = convert_bbox_to_high_dpi(bbox, source_dpi=72, target_dpi=self.dpi)
                    cropped_block = crop_cv_img(page_image, bbox)
                    safe_bbox = '_'.join(str(int(v)) for v in bbox)
                    img_name = f"image_p{page_idx}_b{block_id}_{safe_bbox}.png"
                    img_path = os.path.join(self.cache_dir, img_name)
                    cv2.imwrite(img_path, cropped_block)
                    block["image_path"] = img_path
                
                # ========== 文本块：无需裁剪，但保留原始内容 ==========
                # （mk_blocks_to_markdown 已经处理好拼接）
                
                # 将块添加到结果列表
                all_blocks.append(block)
                if False:
                    # 可视化：在页面上绘制边界框（调试用）
                    cv2.rectangle(page_image, 
                                (int(block["bbox"][0]), int(block["bbox"][1])), 
                                (int(block["bbox"][2]), int(block["bbox"][3])), 
                                (0, 255, 0), 2)
                    # 可选：添加类型标签
                    cv2.putText(page_image, block_type, 
                                (int(block["bbox"][0]), int(block["bbox"][1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                block_id += 1
            
            # 保type存带有边界框的页面图片（用于调试）
            
            if False:
                img_name = f"page_id_{page_idx}_vis.png"
                img_path = os.path.join(self.cache_dir, img_name)
                cv2.imwrite(img_path, page_image)
            
        
        return all_blocks
    
    def _crop_image_from_page(self, page_image: np.ndarray, bbox: List[float],
                              page_idx: int, block_id: int, prefix: str = "crop") -> str:
        """从页面图像数组中裁剪指定区域并保存（保留此方法以备单独调用）"""
        cropped = crop_cv_img(page_image, bbox)
        safe_bbox = '_'.join(str(int(v)) for v in bbox)
        img_name = f"{prefix}_p{page_idx}_b{block_id}_{safe_bbox}.png"
        img_path = os.path.join(self.cache_dir, img_name)
        cv2.imwrite(img_path, cropped)
        return img_path
    
    def close(self) -> None:
        if self.doc:
            self.doc.close()
            self.doc = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        

from typing import List, Dict, Any, Optional, Union
import re
import os

from abc import ABC, abstractmethod
import os
import unicodedata
from fast_langdetect import detect_language

def is_hyphen_at_line_end(line):
    """Check if a line ends with one or more letters followed by a hyphen.

    Args:
    line (str): The line of text to check.

    Returns:
    bool: True if the line ends with one or more letters followed by a hyphen, False otherwise.
    """
    # Use regex to check if the line ends with one or more letters followed by a hyphen
    return bool(re.search(r'[A-Za-z]+-\s*$', line))

if not os.getenv("FTLANG_CACHE"):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    root_dir = os.path.dirname(current_dir)
    ftlang_cache_dir = os.path.join(root_dir, 'resources', 'fasttext-langdetect')
    os.environ["FTLANG_CACHE"] = str(ftlang_cache_dir)
    # print(os.getenv("FTLANG_CACHE"))



def full_to_half_exclude_marks(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters and numbers (FF21-FF3A for A-Z, FF41-FF5A for a-z, FF10-FF19 for 0-9)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)


def full_to_half(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters, numbers and punctuation (FF01-FF5E)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)

def remove_invalid_surrogates(text):
    # 移除无效的 UTF-16 代理对
    return ''.join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def detect_lang(text: str) -> str:

    if len(text) == 0:
        return ""

    text = text.replace("\n", "")
    text = remove_invalid_surrogates(text)

    # print(text)
    try:
        lang_upper = detect_language(text)
    except:
        html_no_ctrl_chars = ''.join([l for l in text if unicodedata.category(l)[0] not in ['C', ]])
        lang_upper = detect_language(html_no_ctrl_chars)

    try:
        lang = lang_upper.lower()
    except:
        lang = ""
    return lang



class DataWriter(ABC):
    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write the data to the file.

        Args:
            path (str): the target file where to write
            data (bytes): the data want to write
        """
        pass

    def write_string(self, path: str, data: str) -> None:
        """Write the data to file, the data will be encoded to bytes.

        Args:
            path (str): the target file where to write
            data (str): the data want to write
        """

        def safe_encode(data: str, method: str):
            try:
                bit_data = data.encode(encoding=method, errors='replace')
                return bit_data, True
            except:  # noqa
                return None, False

        for method in ['utf-8', 'ascii']:
            bit_data, flag = safe_encode(data, method)
            if flag:
                self.write(path, bit_data)
                break

class FileBasedDataWriter(DataWriter):
    def __init__(self, parent_dir: str = '') -> None:
        """Initialized with parent_dir.

        Args:
            parent_dir (str, optional): the parent directory that may be used within methods. Defaults to ''.
        """
        self._parent_dir = parent_dir

    def write(self, path: str, data: bytes) -> None:
        """Write file with data.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            data (bytes): the data want to write
        """
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        if not os.path.exists(os.path.dirname(fn_path)) and os.path.dirname(fn_path) != "":
            os.makedirs(os.path.dirname(fn_path), exist_ok=True)


        with open(fn_path, 'wb') as f:
            f.write(data)

class ContentType:
    IMAGE = 'image'
    TABLE = 'table'
    TEXT = 'text'
    INTERLINE_EQUATION = 'interline_equation'
    INLINE_EQUATION = 'inline_equation'
    EQUATION = 'equation'

class BlockType:
    IMAGE = 'image'
    TABLE = 'table'
    IMAGE_BODY = 'image_body'
    TABLE_BODY = 'table_body'
    IMAGE_CAPTION = 'image_caption'
    TABLE_CAPTION = 'table_caption'
    IMAGE_FOOTNOTE = 'image_footnote'
    TABLE_FOOTNOTE = 'table_footnote'
    TEXT = 'text'
    TITLE = 'title'
    INTERLINE_EQUATION = 'interline_equation'
    LIST = 'list'
    INDEX = 'index'
    DISCARDED = 'discarded'

    # Added in vlm 2.5
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    ALGORITHM = "algorithm"
    REF_TEXT = "ref_text"
    PHONETIC = "phonetic"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"
    
class MakeMode:
    MM_MD = 'mm_markdown'
    NLP_MD = 'nlp_markdown'
    CONTENT_LIST = 'content_list'

default_delimiters = {
    'display': {'left': '$$', 'right': '$$'},
    'inline': {'left': '$', 'right': '$'}
}




class ListLineTag:
    IS_LIST_START_LINE = 'is_list_start_line'
    IS_LIST_END_LINE = 'is_list_end_line'
    

def __is_hyphen_at_line_end(line):
    """Check if a line ends with one or more letters followed by a hyphen.

    Args:
    line (str): The line of text to check.

    Returns:
    bool: True if the line ends with one or more letters followed by a hyphen, False otherwise.
    """
    # Use regex to check if the line ends with one or more letters followed by a hyphen
    return bool(re.search(r'[A-Za-z]+-\s*$', line))


def full_to_half(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters and numbers (FF21-FF3A for A-Z, FF41-FF5A for a-z, FF10-FF19 for 0-9)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)

def has_chinese(text):
    """判断字符串是否包含中文"""
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 基本汉字范围
    return bool(pattern.search(text))

# Constants
default_delimiters = {
    'display': {'left': '$$', 'right': '$$'},
    'inline': {'left': '$', 'right': '$'}
}
delimiters = default_delimiters
display_left_delimiter = delimiters['display']['left']
display_right_delimiter = delimiters['display']['right']
inline_left_delimiter = delimiters['inline']['left']
inline_right_delimiter = delimiters['inline']['right']



    
def get_title_level(block: Dict[str, Any]) -> int:
    """获取标题级别"""
    title_level = block.get('level', 1)
    return max(1, min(title_level, 4))

def escape_special_markdown_char(content):
    """
    转义正文里对markdown语法有特殊意义的字符
    """
    special_chars = ["*", "`", "~", "$"]
    for char in special_chars:
        content = content.replace(char, "\\" + char)

    return content

def normalize_rect(rect):
    """确保矩形坐标为 (x_min, y_min, x_max, y_max) 格式"""
    x1, y1, x2, y2 = rect
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

def merge_bbox(rectangles):
    """
    合并多个矩形区域，返回最小外接矩形
    输入: [(x1,y1,x2,y2), ...]
    输出: (x_min, y_min, x_max, y_max)
    """
    if not rectangles:
        return None
    
    # 初始化极值
    x_min = y_min = float('inf')
    x_max = y_max = -float('inf')
    
    for rect in rectangles:
        norm_rect = normalize_rect(rect)
        x_min = min(x_min, norm_rect[0])
        y_min = min(y_min, norm_rect[1])
        x_max = max(x_max, norm_rect[2])
        y_max = max(y_max, norm_rect[3])
    
    return [x_min, y_min, x_max, y_max]



def merge_para_with_text(para_block, formula_enable=True, img_buket_path=''):
    block_text = ''
    for line in para_block['lines']:
        for span in line['spans']:
            if span['type'] in [ContentType.TEXT]:
                span['content'] = full_to_half_exclude_marks(span['content'])
                block_text += span['content']
    block_lang = detect_lang(block_text)

    para_text = ''
    for i, line in enumerate(para_block['lines']):
        for j, span in enumerate(line['spans']):
            span_type = span['type']
            content = ''
            if span_type == ContentType.TEXT:
                content = span['content']
            elif span_type == ContentType.INLINE_EQUATION:
                content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
            elif span_type == ContentType.INTERLINE_EQUATION:
                if formula_enable:
                    content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
                else:
                    if span.get('image_path', ''):
                        content = f"![]({img_buket_path}/{span['image_path']})"

            content = content.strip()
            if content:

                if span_type == ContentType.INTERLINE_EQUATION:
                    para_text += content
                    continue

                # 定义CJK语言集合(中日韩)
                cjk_langs = {'zh', 'ja', 'ko'}
                # logger.info(f'block_lang: {block_lang}, content: {content}')

                # 判断是否为行末span
                is_last_span = j == len(line['spans']) - 1

                if block_lang in cjk_langs:  # 中文/日语/韩文语境下，换行不需要空格分隔,但是如果是行内公式结尾，还是要加空格
                    if is_last_span and span_type != ContentType.INLINE_EQUATION:
                        para_text += content
                    else:
                        para_text += f'{content} '
                else:
                    # 西方文本语境下 每行的最后一个span判断是否要去除连字符
                    if span_type in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
                        # 如果span是line的最后一个且末尾带有-连字符，那么末尾不应该加空格,同时应该把-删除
                        if (
                                is_last_span
                                and span_type == ContentType.TEXT
                                and is_hyphen_at_line_end(content)
                        ):
                            # 如果下一行的第一个span是小写字母开头，删除连字符
                            if (
                                    i+1 < len(para_block['lines'])
                                    and para_block['lines'][i + 1].get('spans')
                                    and para_block['lines'][i + 1]['spans'][0].get('type') == ContentType.TEXT
                                    and para_block['lines'][i + 1]['spans'][0].get('content', '')
                                    and para_block['lines'][i + 1]['spans'][0]['content'][0].islower()
                            ):
                                para_text += content[:-1]
                            else:  # 如果没有下一行，或者下一行的第一个span不是小写字母开头，则保留连字符但不加空格
                                para_text += content
                        else:  # 西方文本语境下 content间需要空格分隔
                            para_text += f'{content} '
    return para_text


def create_block_data(
    content: str,
    block_type: str,
    bbox: List[float],
    children: Optional[List[Dict[str, Any]]] = None,
    dpi: int = 72
) :
    """创建标准化的块数据结构"""
    return {
        "content": content,
        "type": block_type,
        "bbox": bbox,#convert_bbox_to_high_dpi(bbox, 72, dpi),
        "children": children or []
    }
def process_image_block(
    para_block: Dict[str, Any],
    img_buket_path: str,
    dpi : int,
):
    """处理图片类型块"""
    components = {
        'caption': {'text': '', 'bboxes': []},
        'body': {'text': '', 'bboxes': []},
        'footnote': {'text': '', 'bboxes': []}
    }
    
    has_footnote = any(block['type'] == BlockType.IMAGE_FOOTNOTE for block in para_block['blocks'])
    
    for block in para_block['blocks']:
        if block['type'] == BlockType.IMAGE_CAPTION:
            components['caption']['text'] += merge_para_with_text(block) + '  \n'
            components['caption']['bboxes'].append(block['bbox'])
        elif block['type'] == BlockType.IMAGE_BODY:
            for line in block['lines']:
                for span in line['spans']:
                    if span['type'] == ContentType.IMAGE and span.get('image_path', ''):
                        img_markdown = f"![]({img_buket_path}/{span['image_path']})"
                        components['body']['text'] += img_markdown
            components['body']['bboxes'].append(block['bbox'])
        elif block['type'] == BlockType.IMAGE_FOOTNOTE and has_footnote:
            components['footnote']['text'] += '  \n' + merge_para_with_text(block)
            components['footnote']['bboxes'].append(block['bbox'])
    
    # 构建子块
    children = []
    bboxes = []
    
    for comp_type, comp_data in components.items():
        if comp_data['text']:
            bbox = merge_bbox(comp_data['bboxes'])
            
            children.append({
                "type": f"image_{comp_type}",
                "content": comp_data['text'],
                "bbox": bbox#convert_bbox_to_high_dpi(bbox)
            })
            bboxes.append(bbox)
    
    # 构建主块
    main_content = ''.join([comp['text'] for comp in components.values()])
    return create_block_data(
        content=main_content,
        block_type='image',
        bbox=merge_bbox(bboxes) if bboxes else [],
        children=children,
        dpi=dpi
    )
def process_table_block(
    para_block: Dict[str, Any],
    table_enable: bool,
    img_buket_path: str,
    dpi: int,
):
    """处理表格类型块"""
    components = {
        'caption': {'text': '', 'bboxes': []},
        'body': {'text': '', 'bboxes': [], 'table_bboxes': []},  # 新增table_bboxes用于存储每个表格的实际bbox
        'footnote': {'text': '', 'bboxes': []}
    }
    for block in para_block['blocks']:
        if block['type'] == BlockType.TABLE_CAPTION:
            merge_text = merge_para_with_text(block)
            
            if not has_chinese(merge_text):
                components['caption']['text'] += merge_text
            # components['caption']['text'] += merge_para_with_text(block) + '  \n'
                components['caption']['bboxes'].append(block['bbox'])
        elif block['type'] == BlockType.TABLE_BODY:
            components['body']['bboxes'].append(block['bbox'])
            for line in block['lines']:
                for span in line['spans']:
                    if span['type'] == ContentType.TABLE:
                        # 记录每个表格的实际bbox
                        if 'bbox' in span:
                            components['body']['table_bboxes'].append(span['bbox'])
                        elif 'image_bbox' in span:
                            components['body']['table_bboxes'].append(span['image_bbox'])
                        
                        if table_enable and span.get('html', ''):
                            table_html = f"\n{span['html']}\n"
                            components['body']['text'] += table_html

                        elif span.get('image_path', ''):
                            table_img = f"![]({img_buket_path}/{span['image_path']})"
                            components['body']['text'] += table_img

        elif block['type'] == BlockType.TABLE_FOOTNOTE:
            components['footnote']['text'] += '\n' + merge_para_with_text(block) + '  '
            components['footnote']['bboxes'].append(block['bbox'])
    
    # 构建子块
    children = []
    bboxes = []
    
    for comp_type, comp_data in components.items():
        if comp_data['text']:
            # 对于表格体，如果有单独的表格bbox，则使用它们
            if comp_type == 'body' and comp_data['table_bboxes']:
                bbox = merge_bbox(comp_data['table_bboxes'])
            else:
                bbox = merge_bbox(comp_data['bboxes'])
            if comp_type == 'caption' and  has_chinese(comp_data['text']):
                continue
            else:
                children.append({
                    "type": f"table_{comp_type}",
                    "content": comp_data['text'],
                    "bbox": bbox##convert_bbox_to_high_dpi(bbox, 72, dpi)
                })
            # if comp_type != 'caption':
            bboxes.append(bbox)
    ## 剔除边缘点
    
    # 构建主块
    main_content = ''.join([comp['text'] for comp in components.values()])
    return create_block_data(
        content=main_content,
        block_type='table',
        bbox=merge_bbox(bboxes) if bboxes else [],
        children=children,
        dpi=dpi
    )

def mk_blocks_to_markdown(
    para_blocks: List[Dict[str, Any]],
    table_enable: bool,
    img_buket_path: str = '',
    dpi: int = 72
) -> List[Dict[str, Any]]:
    """将段落块转换为Markdown格式"""
    page_markdown = []
    
    for para_block in para_blocks:
        para_type = para_block['type']
        block_data = None
        if para_type in [
                BlockType.TEXT,
                BlockType.REF_TEXT,
                BlockType.PHONETIC,
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.PAGE_NUMBER,
                BlockType.ASIDE_TEXT,
                BlockType.PAGE_FOOTNOTE,
            ]:

            content = merge_para_with_text(para_block)
            block_data = create_block_data(content, 'text', para_block['bbox'], dpi=dpi)

        elif para_type in [BlockType.DISCARDED]:
            content = '\n' + merge_para_with_text(para_block) + '  '
            block_data = create_block_data(content, 'discarded', para_block['bbox'], dpi=dpi)
            
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            content = f'{"#" * title_level} {merge_para_with_text(para_block)}'
            block_data = create_block_data(content, 'title', para_block['bbox'], dpi= dpi)
            
        elif para_type == BlockType.IMAGE:
            block_data = process_image_block(para_block, img_buket_path, dpi)
            
        elif para_type == BlockType.TABLE:
            block_data = process_table_block(para_block, table_enable, img_buket_path, dpi = dpi)
        
        if block_data and block_data['content'].strip():
            page_markdown.append(block_data)
    return page_markdown

"""Data utility functions."""
import os
import json
import yaml
import re
from typing import List, Dict, Any

def find_mineru_files(folder_path):
    pdf_file = ""
    for file in os.listdir(folder_path):

        if file.endswith('.pdf'):
            pdf_file = os.path.join(folder_path, file)
    return pdf_file, os.path.join(folder_path, 'layout.json')


def load_yaml(file_path: str) -> Dict:
    """Load YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict, file_path: str):
    """Save as YAML file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, file_path: str):
    """Save as JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_schema(schema_path: str) -> Dict:
    """Load schema from JSON or YAML file."""
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            if schema_path.endswith('.json'):
                return json.load(f)
            elif schema_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported schema format")
    except Exception as e:
        print(f"Error loading schema: {e}")
        return {}


def save_results(results: Any, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def filter_non_null_data(extracted_data: Dict) -> Dict:
    """Filter out null values from extracted data."""
    return {k: v for k, v in extracted_data.items() 
            if v not in [None, "null", "", "Null", "N/A"]}
    

def load_markdown(file_path: str) -> str:
    """Load markdown file content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_text(text: str) -> str:
    """Basic text preprocessing: remove excessive whitespace, etc."""
    # Replace multiple newlines with single newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove leading/trailing spaces
    text = text.strip()
    return text

def create_text_blocks(text: str, max_tokens: int = 8000, tokenizer=None) -> List[str]:
    """
    Split text into blocks not exceeding max_tokens.
    Simple heuristic: split by paragraphs and accumulate.
    """
    if tokenizer is None:
        # Rough approximation: tokens ~ words * 1.3
        def approx_tokens(t):
            return len(t.split()) * 1.3
        tokenizer = approx_tokens
    
    paragraphs = text.split('\n\n')
    blocks = []
    current_block = []
    current_len = 0
    
    for para in paragraphs:
        para_len = tokenizer(para)
        if current_len + para_len > max_tokens and current_block:
            blocks.append('\n\n'.join(current_block))
            current_block = []
            current_len = 0
        current_block.append(para)
        current_len += para_len
    
    if current_block:
        blocks.append('\n\n'.join(current_block))
    
    return blocks

def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a single record: extract numeric values from strings, standardize units.
    This is a simplified version; can be expanded.
    """
    cleaned = {}
    for key, value in record.items():
        if value is None:
            cleaned[key] = None
            continue
        # If it's a string, try to extract a number
        if isinstance(value, str):
            # Attempt to find a number (including scientific)
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
            if match:
                try:
                    cleaned[key] = float(match.group())
                except:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        elif isinstance(value, dict) and value.get('_is_extra'):
            # For extra fields, extract numeric value if present
            if 'value' in value and isinstance(value['value'], (int, float)):
                cleaned[key] = value['value']
            else:
                cleaned[key] = value.get('original_value', None)
        else:
            cleaned[key] = value
    return cleaned
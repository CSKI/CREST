import os
import sys
import argparse
import logging
import uuid
from typing import List, Dict

from utils.llm_client import OpenAIClient,GeminiClient
from utils.file_utils import load_yaml, load_markdown, save_json, find_mineru_files, load_json
from utils.mineru import MinerUParser
from algorithms.data_structure import dicts_to_blocks
from algorithms.mllm_audit import audit_by_block_natural
from algorithms.document_extractor_planning import DocumentPlanningExtractor
from algorithms.document_planner import associate_blocks_llm
from algorithms.pipline_tools import identify_experiment_tables_llm, add_confidence_to_records_from_paper, add_evidence_to_records_from_paper

from collections import defaultdict


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


global_data_dir = "cache/"
def get_rec_images_text(records, env):
    
    return 
def main(llm_extractor: OpenAIClient,
         llm_judger: OpenAIClient,
         mllm_judger: OpenAIClient,
         extractor: DocumentPlanningExtractor,
         schema: Dict,
         mineru_json: str,
         pdf_path: str,
         use_table_classifier: bool,
         use_dynamic_context: bool=True,
         use_evidence: bool=False,
         use_confidence: bool=False,
         use_cv: bool=False,
         output_dir: str="output"):


    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    image_cache_dir = os.path.join(output_dir, "images")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    # Initialize LLM client
    
    parser = MinerUParser(pdf_path, mineru_json, cache_dir = image_cache_dir)
    blocks = parser.load_blocks()
    blocks = dicts_to_blocks(blocks)
    
    raw_records = []
    if use_table_classifier:

        if os.path.exists(os.path.join(data_dir, "exp_tables.json")):
            exp_idx = load_json(os.path.join(data_dir, "exp_tables.json"))
        else:
            exp_tables = identify_experiment_tables_llm(blocks, llm_extractor)
        
            print(f"Found {len(exp_tables)} experiment tables")
            exp_idx =  [x.id for x in exp_tables]
    
            save_json(exp_idx, os.path.join(data_dir, "exp_tables.json"))
        exp_tables = [bk for i in exp_idx for bk in blocks if bk.id == i]

        raw_records = []
        
        for exp_table in exp_tables:
            
            if use_dynamic_context:

                if os.path.exists(os.path.join(data_dir, f"{exp_table.id}_linking.json")):
                    table_blocks_idx = load_json(os.path.join(data_dir, f"{exp_table.id}_linking.json"))
                else:
                    table_blocks_idx = associate_blocks_llm(exp_table, blocks, llm_extractor)
                    new_blocks = [block for block in blocks if block.id in table_blocks_idx]
                    save_json(table_blocks_idx, os.path.join(data_dir, f"{exp_table.id}_linking.json"))
                print(f"Found {len(table_blocks_idx)}: {table_blocks_idx} associated blocks for {exp_table.id}"'')
        #         #new_blocks = [block for block in blocks if block.id != exp_table.id]

            if os.path.exists(os.path.join(data_dir, f"{exp_table.id}_records.json")):
                record = load_json(os.path.join(data_dir, f"{exp_table.id}_records.json"))
            else:
               # new_blocks = [block for block in blocks if block.id in exp_idx]
                paper_context = "\n".join([x.content for x in blocks])
                record = extractor.extract(paper_context, exp_table.content)
                raw_records.append(record)
                save_json(record, os.path.join(data_dir, f"{exp_table.id}_records.json"))

            if use_evidence:
                if os.path.exists(os.path.join(data_dir, f"{exp_table.id}_evidence.json")):
                    evidence = load_json(os.path.join(data_dir, f"{exp_table.id}_evidence.json"))
                else:
                    evidence = add_evidence_to_records_from_paper(record, blocks, schema,llm_judger)
                    save_json(evidence, os.path.join(data_dir, f"{exp_table.id}_evidence.json"))

                if os.path.exists(os.path.join(data_dir, f"{exp_table.id}_mllm_judger_result.json")):
                    mllm_judger_result = load_json(os.path.join(data_dir, f"{exp_table.id}_mllm_judger_result.json"))
                else:
                    mllm_judger_result = audit_by_block_natural(record, evidence, blocks, mllm_judger, schema, image_cache_dir)
                    save_json(mllm_judger_result, os.path.join(data_dir, f"{exp_table.id}_mllm_judger_result.json"))
                
            if use_confidence:
                if os.path.exists(os.path.join(data_dir,f"{exp_table.id}_confidence.json")):
                    confidence = load_json(os.path.join(data_dir,f"{exp_table.id}_confidence.json"))
                else:
                    confidence = add_confidence_to_records_from_paper(record, blocks, schema, llm_judger)
                    save_json(confidence, os.path.join(data_dir,f"{exp_table.id}_confidence.json"))
                    
    save_json(raw_records, os.path.join(data_dir, "records.json"))

if __name__ == "__main__":
    
    llm_extractor = OpenAIClient(model_name="deepseek-chat", 
                              base_url="https://api.deepseek.com", 
                              api_key="sk-d6a737a07d9f4a0ebf7ea63997358483")  
    llm_judger = OpenAIClient(model_name="deepseek-reasoner", 
                              base_url="https://api.deepseek.com", 
                              api_key="sk-d6a737a07d9f4a0ebf7ea63997358483")  
    mll_client = OpenAIClient(model_name="qwen-vl-plus",
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                            api_key="sk-2196ed3d775242f3972c72719aff024d")
    # vllm_client = OpenAIClient(model_name="qwen-vl-plus",
    #                         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    #                         api_key="sk-2196ed3d775242f3972c72719aff024d")
    
    # llm_client = GeminiClient(
    #     model_name="gemini-2.5-flash",
    #     api_key="AIzaSyD6fJtEeDWpTv6CFA6UtYdaO7VtS02CY0Q"
    # )
    # llm_client = OpenAIClient(model_name="qwen-plus",
    #                         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    #                         api_key="sk-2196ed3d775242f3972c72719aff024d")
    
    mineru_result = "/Users/wangpeng/Desktop/工作/AlgorithmCode/CCLIP/data/MinerU/"
    # mineru_result = "/Users/wangpeng/Desktop/工作/AlgorithmCode/cclip-realse/CCLLMP-master/test_data/short/"
    output_path = "output/gnimi/" 
    output_path = "output/qwen/" 
    output_path = "/Users/wangpeng/Desktop/工作/AlgorithmCode/cclip-realse/CCLLMP-master/test_data/preditction/no_extra_deep/" 
    output_path = "test_data/preditction/piplines-notice-v3/"
    # 需要处理的文件夹列表
    val_folders = [
        "pdf-9f4f0167-6f3e-4386-8cbd-db18f1808c5b",
        "pdf-6113bf6e-fc67-48fe-9517-afa3aa22c40a",
        "pdf-f060708f-8e41-4b16-9b3e-92ec1145ec02",
        "pdf-8ce0dfbe-ebf1-4359-9c8c-3c45ad3cc5bc",
        "pdf-8415379b-7293-4923-bc48-dfe9f305416b",
        "pdf-20c35206-ef5b-4d5f-9cb1-12bb090ae745",
        "pdf-c6ce544a-d154-48c7-820e-ade154602c46",
        "pdf-6d68a2da-61d1-490a-93e6-5c09e7d824b0",
        "pdf-60f25c14-a062-4ed0-955f-21ee23d18296",
        "pdf-9202dbb0-04eb-460c-8804-2c1d9c413777",
        "pdf-68512bbc-cddb-4579-92de-43c7f4efa890"
    ]

    for file_name in os.listdir(mineru_result):
        if file_name not in val_folders:
            continue
        # if file_name != "pdf-9f4f0167-6f3e-4386-8cbd-db18f1808c5b":
        #     continue
        logger.info(f'{file_name} is processing')

        sub_f = os.path.join(mineru_result, file_name)
        if not os.path.isdir(sub_f):
            continue
        pdf_path, json_path = find_mineru_files(mineru_result + file_name)
        if not os.path.exists(json_path):
            continue
        file_outpath = output_path + file_name
        image_cache_dir = file_outpath + '/images'
        
        if not os.path.exists(image_cache_dir):
            os.makedirs(image_cache_dir, exist_ok=True)
            
        schema = "/Users/wangpeng/Desktop/工作/AlgorithmCode/CCLIP/configs/schema.json"
        schema = load_json(schema)
        extractor = DocumentPlanningExtractor(llm_extractor, schema, chunk_size=5)
        main(
            llm_extractor,
            llm_judger,
            mll_client,
            extractor,
            schema=schema,
            mineru_json=json_path,
            pdf_path=pdf_path,
            use_table_classifier=True,
            use_dynamic_context=False,
            use_evidence=False,
            use_confidence=False,
            use_cv=False,
            output_dir=file_outpath
            )
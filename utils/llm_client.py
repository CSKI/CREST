"""LLM client for API calls."""

import json
import logging
import re
from typing import Any, Optional
from openai import OpenAI
# 在文件顶部添加导入（如果尚未安装 google-genai，需提前安装）
from google import genai
from google.genai import types
import time


logger = logging.getLogger(__name__)

import time
class LLMClient:
    """Base class for LLM clients."""

    def __init__(self, model_name: str, base_url: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)


    def extract_json_from_response(self, response: str) -> Any:
        """Extract JSON from LLM response (could be dict or list)."""
        if not response:
            return None

        # 1. 优先提取 markdown 代码块
        code_block_pattern = r'```(?:json|JSON)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        for candidate in matches:
            parsed = self._try_parse_json(candidate.strip())
            if parsed is not None:
                return parsed

        # 2. 尝试提取最外层的 {...} 或 [...]
        for pattern in (r'(\{.*\})', r'(\[.*\])'):
            # 使用非贪婪但跨越整个字符串的多行匹配
            # 实际需要平衡括号，所以用自定义函数
            balanced = self._extract_balanced_json(response, pattern[1])
            if balanced:
                parsed = self._try_parse_json(balanced)
                if parsed is not None:
                    return parsed

        # 3. 最后尝试直接解析整个字符串
        return self._try_parse_json(response)

    def _extract_balanced_json(self, text: str, open_char: str = '{') -> str:
        """提取最外层平衡的 JSON 片段（支持 {} 或 []）"""
        close_char = '}' if open_char == '{' else ']'
        start = text.find(open_char)
        if start == -1:
            return ""
        stack = 0
        for i in range(start, len(text)):
            if text[i] == open_char:
                stack += 1
            elif text[i] == close_char:
                stack -= 1
                if stack == 0:
                    return text[start:i+1]
        return ""

    def _try_parse_json(self, s: str) -> Any:
        """尝试解析 JSON，自动修复常见问题"""
        if not s:
            return None
        s = s.strip()
        # 移除控制字符（除了必要的空白）
        s = ''.join(ch for ch in s if ord(ch) >= 32 or ch in '\n\r\t')
        # 尝试直接解析
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        # 修复尾随逗号（常见于 LLM 生成的 JSON）
        try:
            s_fixed = re.sub(r',\s*}', '}', s)   # {...,} -> {...}
            s_fixed = re.sub(r',\s*]', ']', s_fixed) # [...,] -> [...]
            return json.loads(s_fixed)
        except json.JSONDecodeError:
            pass

        # 尝试将单引号替换为双引号（不完美但常用）
        try:
            # 仅替换成对单引号包围的字符串，避免破坏已存在的双引号
            s_fixed = re.sub(r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', s)
            return json.loads(s_fixed)
        except json.JSONDecodeError:
            pass

        # 如果仍然失败，用 ast.literal_eval 尝试（更宽松）
        try:
            import ast
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass

        logger.error(f"Failed to parse JSON after all attempts: {s[:200]}")
        return None

    def call(self, prompt: str, **kwargs) -> Any:
        """Abstract method for calling LLM."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def call(self, prompt: str, temperature: float = 0.1, max_tokens: int = 8190) -> Any:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0
              #  response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            #logger.debug(f"LLM response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return ""
    def call_with_history(self, messages, temperature: float = 0.1, max_tokens: int = 8190) -> Any:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False
              #  response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            #logger.debug(f"LLM response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return ""
    def call_with_evidence(self, messages, temperature: float = 0.1, max_tokens: int = 8190) -> Any:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False
              #  response_format={ "type": "json_object" }
            )
            content = response.choices[0].message
            #logger.debug(f"LLM response: {content[:100]}...")
            return content
        except Exception as e:  
            logger.error(f"OpenAI API call failed: {e}")
            return None
        
    def call_with_image(self, prompt: str, image_url: str, temperature=0.0, max_tokens=4096):
        # response = self.client.chat.completions.create(
        #     model=self.model_name,  # 或 gpt-4o
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": prompt},
        #                 {"type": "image_url", "image_url": {"url": image_url}}
        #             ]
        #         }
        #     ]
        # )
        import base64
        def encode_image_to_base64(image_path: str) -> str:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        #image_path = "test_data/preditction/piplines/pdf-9f4f0167-6f3e-4386-8cbd-db18f1808c5b/images/table_p6_b28_69_153_526_342.png"
        image_url = f"data:image/png;base64,{encode_image_to_base64(image_url)}"
        response = self.client.chat.completions.create(
            model="qwen-vl-plus",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[{"role": "user",
                       "content": [
                            {"type": "image_url",
                            "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                    ]}]
    )   
        #print(response.choices[0].message.content)
        return response.choices[0].message.content

    def call_with_usage(self, prompt: str, **kwargs):
        """返回 (content, usage) 元组，usage 可能为 None"""
        
        try:
            st = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            content = response.choices[0].message.content
            usage = response.usage
            cost = time.time() - st
            logger.info(f"LLM call success, cost: {cost}s, usage: {usage}")
            return content, [usage.completion_tokens,usage.prompt_tokens, usage.total_tokens,cost]
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "", ""
        
class MockLLMClient(LLMClient):
    """Mock client for testing."""

    def call(self, prompt: str, **kwargs) -> Any:
        """Mock call for testing."""
        return """```json
[
  {
    "catalyst_name": "Test Catalyst",
    "yield_g": 5.2,
    "additional_items": [
      {
        "item_name": "test_field",
        "define": "Test field definition",
        "value": "test_value",
        "suggested_data_type": "String"
      }
    ]
  }
]
```"""

# 在 LLMClient 的子类区域添加 GeminiClient
class GeminiClient(LLMClient):
    """Google Gemini API client."""

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        # Gemini 的 base_url 通常不需要显式设置，但保留参数以兼容父类
        super().__init__(model_name, base_url, api_key)
        # 初始化 Gemini 客户端（使用 API Key）
        self.client = genai.Client(api_key=self.api_key)

    def call(self, prompt: str, temperature: float = 0.0, **kwargs) -> Any:
        """调用 Gemini API，返回文本内容。"""
        try:
            # 构建请求配置
            config = types.GenerateContentConfig(
                temperature=temperature,
                # 可在此添加其他支持参数，如 top_p, top_k 等
                **{k: v for k, v in kwargs.items() if k in ['top_p', 'top_k', 'stop_sequences', 'system_instruction']}
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            # 提取文本内容
            content = response.text
            logger.debug(f"Gemini response: {content[:500]}...")
            return content
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return None

    def call_with_image(self, prompt: str, image_base64: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """多模态调用：文本 + 图片（base64编码）。"""
        try:
            # Gemini 支持 inline data 或上传文件，这里使用 inline base64
            # 注意：base64 字符串需确保不包含 data:image/... 前缀
            from PIL import Image
            import io
            import base64

            # 解码 base64 为图片字节
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini call_with_image failed: {e}")
            return ""

    def call_with_usage(self, prompt: str, **kwargs) -> tuple:
        """返回 (content, usage_info) 元组，usage_info 格式与 OpenAIClient 兼容。"""
        try:
            start_time = time.time()
            config = types.GenerateContentConfig(
                temperature=kwargs.get('temperature', 0.1),
                max_output_tokens=kwargs.get('max_tokens', 8192),
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            content = response.text
            cost = time.time() - start_time

            # 提取 Gemini 的使用统计（如果可用）
            # Gemini SDK 的 response 可能有 usage_metadata 属性
            usage_tokens = [0, 0, 0]  # [completion_tokens, prompt_tokens, total_tokens]
            if hasattr(response, 'usage_metadata'):
                prompt_tokens = response.usage_metadata.prompt_token_count
                candidates_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                usage_tokens = [candidates_tokens, prompt_tokens, total_tokens]
            logger.info(f"Gemini call success, cost: {cost:.2f}s, usage: {usage_tokens}")
            return content, usage_tokens + [cost]
        except Exception as e:
            logger.error(f"Gemini call_with_usage failed: {e}")
            return "", ""
        
if __name__ == "__main__":
    # 简单测试
    # 初始化 Gemini 客户端
    gemini = GeminiClient(
        model_name="gemini-2.5-flash",
        api_key="AIzaSyD6fJtEeDWpTv6CFA6UtYdaO7VtS02CY0Q"
    )

    # # 普通调用
    # response = gemini.call("给我讲个关于程序员的笑话。", temperature=0.8)
    # print(response)

    # 带用量统计的调用
    content, usage = gemini.call_with_usage("解释什么是递归", temperature=0.2)
    print(content, usage)
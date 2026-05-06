import warnings
warnings.filterwarnings("ignore")

import httpx
import torch

class OllamaEmbedder:
    def __init__(self, LLM_embedder_type, base_url="http://127.0.0.1:11434"):
        self.LLM_embedder_type = LLM_embedder_type
        self.base_url = f"{base_url}/api/embed"
        
        # 在初始化时自动探测嵌入维度
        # self.emb_dim = self._probe_dimension()
        # print(f"--- [OllamaEmbedder] ---: Model '{LLM_embedder_type}' detected with dimension: {self.emb_dim}")

    @classmethod
    def _probe_dimension(cls, LLM_embedder_type):
        """
        向 Ollama 发送一个微型测试请求，以确定模型的向量维度
        """
        payload = {
            "model": LLM_embedder_type,
            "input": "probe" 
        }
        try:
            # 这里的超时可以设置短一点，因为只是初始化检查
            with httpx.Client(timeout=30.0) as client:
                response = client.post("http://127.0.0.1:11434/api/embed", json=payload)
                response.raise_for_status()
                embeddings = response.json().get("embeddings", [])
                if embeddings:
                    return len(embeddings[0])
                else:
                    raise ValueError("No embeddings returned from Ollama.")
        except Exception as e:
            print(f"Warning: Failed to probe dimension for {LLM_embedder_type}: {e}")
            # 回退到默认值（DeepSeek-V2-Lite 通常是 4096）
            return False

    @torch.no_grad()
    def get_tactical_directive(self, prompt, device):
        payload = {
            "model": self.LLM_embedder_type,
            "input": prompt
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.base_url, json=payload)
                response.raise_for_status()
                embedding_list = response.json().get("embeddings", [])

                return torch.tensor(embedding_list[0], dtype=torch.float32).to(device)

        except Exception as e:
            # 在实际运行中，如果 API 出错，返回对应维度的零向量
            return self._get_zero_tensor(device)

    # 保持原有接口兼容性
    def to(self, device): return self
    def eval(self): return self
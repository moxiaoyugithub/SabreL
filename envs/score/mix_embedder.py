import sys
import json
import asyncio
import torch
import subprocess
import pickle
import struct
from concurrent.futures import ThreadPoolExecutor
from logs.logger import logger

class MixEmbedderCalculator:
    """对应原来的 MixSimilarityCalculator，负责管理模型子进程"""
    def __init__(self, target_model_types, device='auto', rank=0):
        self.device = device
        self.target_model_types = target_model_types
        self.proc_dict = {}
        
        # 启动常驻子进程
        for model_type in self.target_model_types:
            # 注意：此处调用的是我们刚才写的 embedding.py
            cmd = f"conda run -n {model_type} --no-capture-output python envs/score/{model_type}/scripts/embedding.py -d {self.device} -r {rank}"
            self.proc_dict[model_type] = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                stderr=sys.stderr, text=True, shell=True, bufsize=1
            )

    def _ask_worker(self, model_type, b1, a1, n1):
        """与子进程进行单次管道通信，获取 Base64 编码的向量"""
        proc = self.proc_dict.get(model_type)
        if not proc or proc.poll() is not None:
            return (model_type, None)
        try:
            task = json.dumps({"b1": b1, "a1": a1, "n1": n1})
            proc.stdin.write(task + "\n")
            proc.stdin.flush()
            
            line = proc.stdout.readline().strip()
            if line.startswith("RESULT:"):
                res = line.replace("RESULT:", "")
                if res.startswith("ERROR:"):
                    print(f"DEBUG: Worker {model_type} reported error: {res}") # 打印出详细的堆栈
                    return (model_type, None)
                # 从 Base64 还原 numpy 向量
                import base64
                vec = pickle.loads(base64.b64decode(res))
                return (model_type, vec)
        except Exception as e:
            logger.error(f"Error in {model_type} worker: {e}")
        return (model_type, None)

    def get_embeddings(self, binary_path, function_address, function_name):
        """并行获取该函数在所有模型下的嵌入"""
        tasks = [(mt, binary_path, function_address, function_name) for mt in self.target_model_types]
        embedding_dict = {}
        with ThreadPoolExecutor(max_workers=len(self.target_model_types)) as executor:
            futures = [executor.submit(self._ask_worker, *t) for t in tasks]
            for f in futures:
                mt, vec = f.result()
                embedding_dict[mt] = vec
        return embedding_dict

class MixEmbedderServer:
    """长效后端服务器，监听 Socket 请求"""
    def __init__(self, device, env_num, target_model_types, host='127.0.0.1', port=6000):
        self.host = host
        self.port = port
        # 为每个并行环境启动一套模型
        self.calculators = [MixEmbedderCalculator(target_model_types, device=device, rank=rank) for rank in range(env_num)]
        self.thread_executor = ThreadPoolExecutor(max_workers=env_num * 4)

    async def handle_client(self, reader, writer):
        try:
            # 协议：先读 4 字节长度，再读 Payload
            header = await reader.readexactly(4)
            length = struct.unpack('!I', header)[0]
            data = await reader.readexactly(length)
            
            params = json.loads(data.decode())
            rank = params.get('rank', 0)
            
            # 在线程池中执行耗时的模型推理
            loop = asyncio.get_running_loop()
            embeds = await loop.run_in_executor(
                self.thread_executor,
                self.calculators[rank].get_embeddings,
                params['b1'], params['a1'], params['n1']
            )
            
            # 返回结果（由于包含向量，继续使用 pickle 序列化以保证精度）
            response = {"status": "ok", "embeddings": embeds}
            resp_data = pickle.dumps(response)
            writer.write(struct.pack('!I', len(resp_data)) + resp_data)
        except Exception as e:
            err_resp = pickle.dumps({"status": "error", "message": str(e)})
            writer.write(struct.pack('!I', len(err_resp)) + err_resp)
        
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(f'[MixEmbedderServer]: Started on {self.host}:{self.port}')
        async with server:
            await server.serve_forever()

class MixEmbedderClient:
    def __init__(self, rank, host='127.0.0.1', port=6000):
        self.host = host
        self.port = port
        self.rank = rank

    async def get_embeddings(self, b1, a1, n1):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        
        # 发送请求
        payload = json.dumps({"b1": b1, "a1": a1, "n1": n1, "rank": self.rank}).encode()
        writer.write(struct.pack('!I', len(payload)) + payload)
        await writer.drain()
        
        # 读取响应
        resp_header = await reader.readexactly(4)
        resp_len = struct.unpack('!I', resp_header)[0]
        resp_data = await reader.readexactly(resp_len)
        
        response = pickle.loads(resp_data)
        writer.close()
        await writer.wait_closed()
        
        if response['status'] == 'ok':
            return response['embeddings']
        else:
            raise Exception(response['message'])
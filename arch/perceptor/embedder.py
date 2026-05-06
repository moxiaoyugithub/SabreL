import warnings
warnings.filterwarnings("ignore")

import asyncio
import json
import torch
import torch.nn as nn
import multiprocessing as mp

from arch.perceptor.ollama_embedder import OllamaEmbedder

from arch.perceptor.palmtree_pre_trained_model import eval_utils as utils

from logs.logger import logger

class MixEmbedder(nn.Module):
    def __init__(self, PalmTree_embedder_path, PalmTree_vocab_path, LLM_embedder_type, device='auto'):
        super(MixEmbedder, self).__init__()
        if device == 'auto':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.PalmTree_embedder = self.load_PalmTree_embedder(PalmTree_embedder_path, PalmTree_vocab_path)
        self.embedding_type = 'PalmTree'
        
        if LLM_embedder_type:
            self.LLM_embedder = self.load_LLM_embedder(LLM_embedder_type)
            self.embedding_type = 'combine'
    
    @staticmethod
    def worker_loop(pipe, PalmTree_args, LLM_embedder_type, device):
        """
        子进程驻留方法：初始化模型并进入死循环等待任务
        """
        # 1. 独立初始化模型（每个进程一份显存）
        embedder = MixEmbedder(*PalmTree_args, LLM_embedder_type, device=device)
        logger.info(f"[MixEmbedder]: Worker process on {device} initialized and waiting...", file_only=True)

        while True:
            try:
                # 2. 从管道阻塞式读取任务
                task = pipe.recv()
                if task is None: break  # 退出信号

                raw_data = task['raw_data']
                remaining_budget = task['remaining_budget']
                similarity_details = task['similarity_details']
                kwargs = task['kwargs']

                with torch.no_grad():
                    res = embedder.embedding(raw_data, remaining_budget, similarity_details, **kwargs)
                    # 序列化结果回传
                    response = {k: v.cpu().tolist() if v is not None else None for k, v in res.items()}
                
                # 3. 将结果塞回管道
                pipe.send({"status": "ok", "data": response})
            
            except EOFError:
                # 管道被主进程关闭，优雅退出
                break
                
            except Exception as e:
                # 只有在管道还活着的时候才发错误信息
                try:
                    pipe.send({"status": "error", "message": str(e)})
                except:
                    break
    
    def load_LLM_embedder(self, LLM_embedder_type):
        LLM_embedder = OllamaEmbedder(LLM_embedder_type=LLM_embedder_type)
        LLM_embedder.eval()
        LLM_embedder.to(self.device)
        return LLM_embedder
    
    # PalmTree_embedder_path="./palmtree/transformer.ep19"
    # vocab_path="./palmtree/vocab"
    def load_PalmTree_embedder(self, PalmTree_embedder_path, vocab_path):
        palmtree_embedder = utils.UsableTransformer(model_path=PalmTree_embedder_path, vocab_path=vocab_path)
        palmtree_embedder.eval()
        palmtree_embedder.to(self.device)
        return palmtree_embedder
    
    @classmethod
    def get_output_dim(cls, LLM_embedder_type):
        output_dim = {
            'function_PalmTree_embedding_dim': 128, 
            'function_LLM_embedding_dim': OllamaEmbedder._probe_dimension(LLM_embedder_type)
        }
        if LLM_embedder_type == None:
            output_dim = {'function_PalmTree_embedding_dim': 128, 'function_LLM_embedding_dim': None}
        print('get output_dim:', output_dim)
        return output_dim
    
    def prompts_gen(self, function_LLM_input, remaining_budget=None, similarity_details=None):
        # 1. 处理整个字典为 None 的情况
        if similarity_details is None:
            similarity_details_str = "Unknown"
        else:
            # 2. 处理字典内部值为 None 的情况（修复点）
            similarity_details_str = ", ".join([
                f"{k}: {v:.4f}" if v is not None else f"{k}: N/A" 
                for k, v in similarity_details.items()
            ])

        # 3. 顺便防御一下 remaining_budget 可能为 None 的情况
        budget_val = remaining_budget if remaining_budget is not None else "Unknown"
        
        prompts = f"""### ROLE: BINARY OBFUSCATION STRATEGIST
TASK: Analyze ASM and Similarity to guide an RL Agent.

### RESOURCES
REMAINING BUDGET: {budget_val} instructions.

ACTIONS: [Split (cost 1 inst), Opaque (cost 15~30 insts), Junk (cost 1~2 insts)]

### INPUT DATA
METRICS: {similarity_details_str}
ASM_CODE:
{function_LLM_input}

### REQUIRED STRUCTURED OUTPUT
Observation: <one sentence analysis of current vulnerability>
Bias: <prioritized action from the 3 choices>
Target: <specific block or instruction pattern>
Reasoning: <how this action drops the threat score>

### STRATEGY:"""
        return prompts
    
    def to_LLM_input(self, function_mix_embedder_input):
        function_LLM_input = ""
        for block_idx, block in enumerate(function_mix_embedder_input):
            function_LLM_input += f'Block_{block_idx}:\n'
            for inst in block:
                function_LLM_input += inst
                function_LLM_input += '\n'
            function_LLM_input += '\n'
        return function_LLM_input
    
    def to_palmtree_input(self, function_mix_embedder_input):
        function_palmtree_input = []
        for block in function_mix_embedder_input:
            processed_block = []
            for inst in block:
                p_inst = inst.replace(',', ' ').replace('[', ' [ ').replace(']', ' ] ')
                processed_block.append(p_inst)
            function_palmtree_input.append(processed_block)
        return function_palmtree_input
    
    def LLM_embedding(self, function_mix_embedder_input, remaining_budget, similarity_details):
        function_LLM_input = self.to_LLM_input(function_mix_embedder_input)
        function_LLM_input = self.prompts_gen(function_LLM_input, remaining_budget, similarity_details)
        function_LLM_embedding = self.LLM_embedder.get_tactical_directive(function_LLM_input, device=self.device)
        return function_LLM_embedding
    
    def PalmTree_embedding(self, function_mix_embedder_input, max_instructions, max_blocks):
        """
        参数:
            function: 包含函数信息的对象
            PalmTree_embedder: PalmTree嵌入器
            max_instructions: 每个基本块的最大指令数
            max_blocks: 函数的最大基本块数
            decoder: 解码器对象
            device: 计算设备
        
        返回:
            function_PalmTree_embedding: 形状为 (max_blocks, max_instructions, embedding_dim) 的嵌入张量
            function_PalmTree_mask: 形状为 (max_blocks, max_instructions) 的mask张量，1表示有效位置
        """
        # 1. 获取函数的PalmTree输入格式
        function_palmtree_input = self.to_palmtree_input(function_mix_embedder_input)
        
        # 限制实际处理的基本块数量
        actual_num_blocks = min(len(function_palmtree_input), max_blocks)
        
        # 2. 处理每个基本块
        block_embeddings_list = []
        valid_lengths_per_block = []  # 记录每个基本块的有效指令数
        
        for block_idx in range(actual_num_blocks):
            block_instructions = function_palmtree_input[block_idx]
            # 获取基本块的原始嵌入
            raw_block_embedding = self.PalmTree_embedder.encode(block_instructions, self.device)
            embedding_dim = raw_block_embedding.shape[1]
            
            # 处理截断或填充
            if len(raw_block_embedding) > max_instructions:
                # 截断：只取前max_instructions条指令
                block_embedding = raw_block_embedding[:max_instructions]
                valid_length = max_instructions
            else:
                # 填充：将嵌入扩展到max_instructions长度
                block_embedding = torch.zeros((max_instructions, embedding_dim), device=self.device)
                block_embedding[:len(raw_block_embedding)] = raw_block_embedding
                valid_length = len(raw_block_embedding)
            
            block_embeddings_list.append(block_embedding.unsqueeze(0))  # 增加批次维度
            valid_lengths_per_block.append(valid_length)
        
        # 3. 将所有基本块堆叠起来
        function_PalmTree_embedding = torch.cat(block_embeddings_list, dim=0)
        
        # 4. 对基本块维度进行填充（如果需要）
        current_blocks = function_PalmTree_embedding.shape[0]
        if current_blocks < max_blocks:
            padding_shape = (max_blocks - current_blocks, max_instructions, embedding_dim)
            padding = torch.zeros(padding_shape, device=self.device)
            function_PalmTree_embedding = torch.cat([function_PalmTree_embedding, padding], dim=0)
            # 为填充的基本块添加有效长度0
            valid_lengths_per_block.extend([0] * (max_blocks - current_blocks))
        
        # 5. 创建mask张量
        function_PalmTree_mask = torch.zeros((max_blocks, max_instructions), device=self.device)
        for block_idx, valid_length in enumerate(valid_lengths_per_block):
            if valid_length > 0:
                function_PalmTree_mask[block_idx, :valid_length] = 1
        
        return function_PalmTree_embedding, function_PalmTree_mask
    
    # 作为感知模块时使用的嵌入函数
    def embedding(self, function_mix_embedder_input, remaining_budget, similarity_details, max_instructions, max_blocks):
        if self.embedding_type == 'PalmTree':
            function_PalmTree_embedding, function_PalmTree_mask = self.PalmTree_embedding(function_mix_embedder_input, max_instructions, max_blocks)
            return {
                "function_PalmTree_embedding": function_PalmTree_embedding,
                "function_PalmTree_mask": function_PalmTree_mask
            }
        
        # 结合模式
        else:
            function_PalmTree_embedding, function_PalmTree_mask = self.PalmTree_embedding(function_mix_embedder_input, max_instructions, max_blocks)
            function_LLM_embedding = self.LLM_embedding(function_mix_embedder_input, remaining_budget, similarity_details)
            return {
                "function_PalmTree_embedding": function_PalmTree_embedding,
                "function_PalmTree_mask": function_PalmTree_mask,
                "function_LLM_embedding": function_LLM_embedding
            }

class MixEmbedderServer:
    def __init__(self, PalmTree_embedder_path, PalmTree_vocab_path, LLM_embedder_type, device, env_num, host='127.0.0.1', port=7000):
        self.host = host
        self.port = port
        self.env_num = env_num
        self.pipes = {}  # 存储 rank -> pipe_handle
        self.processes = []
        self.device = device

        # 为每个 rank 启动一个常驻进程
        palm_args = (PalmTree_embedder_path, PalmTree_vocab_path)

        for rank in range(env_num):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=MixEmbedder.worker_loop, 
                args=(child_conn, palm_args, LLM_embedder_type, device)
            )
            p.daemon = True
            p.start()
            self.pipes[rank] = parent_conn
            self.processes.append(p)

    async def handle_client(self, reader, writer):
        try:
            # 解决 TCP 粘包：读取直到连接关闭或数据完整
            raw_payload = await reader.read() 
            if not raw_payload: return
            
            req = json.loads(raw_payload.decode())
            rank = req.get('rank', 0)
            
            # 路由：将任务灌入对应 rank 的管道
            if rank in self.pipes:
                pipe = self.pipes[rank]
                
                # 由于 Pipe.send/recv 是阻塞的，我们扔进线程池执行
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self._communicate_with_pipe, pipe, req)
                
                response = result
            else:
                response = {"status": "error", "message": f"Invalid rank {rank}"}

        except Exception as e:
            response = {"status": "error", "message": str(e)}

        writer.write(json.dumps(response).encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    def _communicate_with_pipe(self, pipe, req):
        """同步方法：负责与子进程管道交互"""
        pipe.send({
            "raw_data": req['raw_data'], 
            "remaining_budget": req['remaining_budget'], 
            "similarity_details": req['similarity_details'], 
            "kwargs": req['kwargs']
        })
        return pipe.recv()

    async def run(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(f'[MixEmbedderServer]: start on {self.device}: {self.host}:{self.port}')
        async with server:
            await server.serve_forever()
    
    def shutdown(self):
        for pipe in self.pipes.values():
            try:
                pipe.send(None)
            except:
                pass
        for p in self.processes:
            p.join(timeout=1)

class MixEmbedderClient:
    def __init__(self, rank, host='127.0.0.1', port=7000):
        self.rank = rank
        self.host = host
        self.port = port

    async def _send_request(self, raw_data, remaining_budget=None, similarity_details=None, **kwargs):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        payload = {
            "rank": self.rank, 
            "raw_data": raw_data, 
            "remaining_budget": remaining_budget, 
            "similarity_details": similarity_details, 
            "kwargs": kwargs
        }
        writer.write(json.dumps(payload).encode())
        await writer.drain()
        writer.write_eof() # 告诉服务器发送完毕，触发 reader.read()

        response_data = await reader.read()
        response = json.loads(response_data.decode())
        
        writer.close()
        await writer.wait_closed()
        
        if response['status'] == 'ok':
            return response['data']
        else:
            raise Exception(response.get('message', 'Unknown error'))
    
    async def embedding(self, function_mix_embedder_input, remaining_budget, similarity_details, max_instructions, max_blocks):
        # 客户端仅发送最原始的指令列表，减少传输开销
        return await self._send_request(
            function_mix_embedder_input, 
            remaining_budget, 
            similarity_details, 
            max_instructions=max_instructions, 
            max_blocks=max_blocks
        )
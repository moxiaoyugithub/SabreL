import warnings
warnings.filterwarnings("ignore")

import sys
import json
import asyncio
import torch
import subprocess
import pickle
import base64
import struct

from concurrent.futures import ThreadPoolExecutor

from logs.logger import logger

# 新的加载方式目的是做到只读不写，通过标准输入输出流传递反汇编之后的结果

# 从一个二进制文件开始，调用process_bin.py内的代码反汇编，但是不保存，而是直接转化为对应的sim_ir

# 调用compare.py，把sim_ir输入sim模型

class MixSimilarityCalculator:
    def __init__(self, target_model_types, device='auto', rank=0):
        if device == 'auto':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.target_model_types = target_model_types
        # self.target_model_types = ['asm2vec', 'BinCola', 'CLAP', 'jTrans', 'safe']
        #self.target_model_types = ['asm2vec']
        #self.target_model_types = ['BinCola']
        #self.target_model_types = ['CLAP']
        #self.target_model_types = ['jTrans']
        #self.target_model_types = ['safe']
        #self.target_model_types = ['BinCola', 'safe']       # 简单课程
        #self.target_model_types = ['BinCola', 'CLAP', 'jTrans', 'safe']        # 复杂课程
        
        # 启动常驻进程，加载模型并进入任务监听循环
        self.proc_dict = {}
        for model_type in self.target_model_types:
            cmd = f"conda run -n {model_type} --no-capture-output python envs/score/{model_type}/scripts/compare.py -d {self.device} -r {rank}"
            self.proc_dict[model_type] = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=sys.stderr, 
                text=True, 
                shell=True, 
                bufsize=1
            )
    
    def _ask_worker(self, model_type, b1, a1, n1, b2, a2, n2):
        """单次对话逻辑"""
        proc = self.proc_dict.get(model_type)
        if not proc or proc.poll() is not None:
            return (model_type, None)

        try:
            task = json.dumps({"b1": b1, "a1": a1, "n1": n1, "b2": b2, "a2": a2, "n2": n2})
            proc.stdin.write(task + "\n")
            proc.stdin.flush()
            
            while True:
                line = proc.stdout.readline()
                if not line: break
                line = line.strip()
                if line.startswith("RESULT:"):
                    res = line.replace("RESULT:", "")
                    if res.startswith("ERROR:"): return (model_type, None)
                    return (model_type, float(res))
        except Exception as e:
            print(e)
            return (model_type, None)

    def compare(self, binary1_path, function_address1, function_name1, binary2_path, function_address2, function_name2, mode='avg'):
        # 使用线程池并行向 4 个常驻进程发请求
        # 线程共享 proc_dict，且 Popen 的管道在单次 write/readline 下是线程安全的
        tasks = [
            (model_type, binary1_path, function_address1, function_name1, binary2_path, function_address2, function_name2)
            for model_type in self.target_model_types
        ]
        
        sim_dict = {}
        with ThreadPoolExecutor(max_workers=len(self.target_model_types)) as executor:
            # 提交任务
            futures = [executor.submit(self._ask_worker, *t) for t in tasks]
            for f in futures:
                mt, val = f.result()
                sim_dict[mt] = val
        
        # (avg/max)
        return self._summarize(sim_dict, mode)
    
    def _summarize(self, sim_dict, mode='avg'):
        valid_vals = [v for v in sim_dict.values() if v is not None]
        if not valid_vals: return 0, sim_dict
        if mode == 'avg': res = sum(valid_vals) / len(valid_vals)
        elif mode == 'max': res = max(valid_vals)
        return res, sim_dict

    def shutdown(self):
        """关闭所有常驻进程"""
        for model, proc in self.proc_dict.items():
            try:
                proc.stdin.write("EXIT\n")
                proc.stdin.flush()
                proc.terminate()
            except:
                pass

# 服务器
class MixSimilarityServer:
    def __init__(self, device, env_num, target_model_types, host='127.0.0.1', port=6000):
        self.host = host
        self.port = port
        self.target_model_types = target_model_types
        # 为每个 rank 预先启动好一组常驻进程 (BinCola, CLAP, etc.)
        self.calculators = [MixSimilarityCalculator(target_model_types, device=device, rank=rank) for rank in range(env_num)]
        # 使用 ThreadPoolExecutor 处理阻塞的管道读取，而不是进程池
        self.thread_executor = ThreadPoolExecutor(max_workers=env_num * 4)
        self.device = self.calculators[0].device

    async def handle_client(self, reader, writer):
        try:
            # 增加读取长度以防 JSON 较长
            data = await reader.read(8192)
            if not data: return
            
            params = json.loads(data.decode())
            rank = params.get('rank', 0)
            
            # 安全检查
            if rank >= len(self.calculators):
                raise ValueError(f"Invalid rank {rank}, max is {len(self.calculators)-1}")

            # 找到对应 rank 的计算器
            calculator = self.calculators[rank]
            loop = asyncio.get_running_loop()

            # 将 compare 逻辑丢进线程池执行（它是 I/O 密集型：写管道 -> 读管道）
            # 这不会阻塞异步 Server 的事件循环
            sim_result, sim_dict = await loop.run_in_executor(
                self.thread_executor,
                calculator.compare,
                params['b1'], params['a1'], params['n1'], params['b2'], params['a2'], params['n2'],
                params['mode']
            )
            
            response = {"status": "ok", "sim_result": sim_result, "sim_dict": sim_dict}
        except Exception as e:
            response = {"status": "error", "message": str(e)}

        writer.write(json.dumps(response).encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(f'[MixSimilarityServer]: using models {self.target_model_types}, start on {self.device}: {self.host}:{self.port}')
        async with server:
            await server.serve_forever()

# 客户端类
class MixSimilarityClient:
    def __init__(self, rank, host='127.0.0.1', port=6000):
        self.host = host
        self.port = port
        self.rank = rank

    async def compare(self, b1, a1, n1, b2, a2, n2, mode='avg'):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        
        payload = {
            "b1": b1, "a1": a1, "n1": n1, "b2": b2, "a2": a2, "n2": n2, 
            "mode": mode, "rank": self.rank
        }
        
        writer.write(json.dumps(payload).encode())
        await writer.drain()
        
        data = await reader.read(8192)
        response = json.loads(data.decode())
        writer.close()
        await writer.wait_closed()
        
        if response['status'] == 'ok':
            return response['sim_result'], response['sim_dict']
        else:
            raise Exception(response['message'])

# ============================================ 影子模式 ===================================================
class ShadowMixSimilarityCalculator:
    def __init__(self, target_model_types, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.target_model_types = target_model_types
        #self.target_model_types = ['asm2vec']
        #self.target_model_types = ['safe']
        #self.target_model_types = ['asm2vec', 'CLAP', 'safe']       # 简单课程
        #self.target_model_types = ['CLAP', 'safe']         # 复杂课程
        
        # 启动常驻进程，加载模型并进入任务监听循环
        self.proc_dict = {}
        for model_type in self.target_model_types:
            cmd = f"conda run -n {model_type} --no-capture-output python envs/score/{model_type}/scripts/shadow_compare.py -d {self.device}"
            self.proc_dict[model_type] = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=sys.stderr, 
                text=True, 
                shell=True, 
                bufsize=1
            )
    
    def _ask_worker(self, model_type, asm1, asm2):
        """单次对话逻辑"""
        proc = self.proc_dict.get(model_type)
        if not proc or proc.poll() is not None:
            return (model_type, None)

        try:
            task = json.dumps({'asm1': asm1, 'asm2': asm2})
            proc.stdin.write(task + "\n")
            proc.stdin.flush()
            
            while True:
                line = proc.stdout.readline()
                if not line: break
                line = line.strip()
                if line.startswith("RESULT:"):
                    res = line.replace("RESULT:", "")
                    if res.startswith("ERROR:"): return (model_type, None)
                    return (model_type, float(res))
        except Exception:
            return (model_type, None)

    def compare(self, asm1, asm2, mode='avg'):
        # 使用线程池并行向 4 个常驻进程发请求
        # 线程共享 proc_dict，且 Popen 的管道在单次 write/readline 下是线程安全的
        tasks = [
            (model_type, asm1.to_shadow_mix_similarity_input(model_type), asm2.to_shadow_mix_similarity_input(model_type))
            for model_type in self.target_model_types
        ]
        
        sim_dict = {}
        with ThreadPoolExecutor(max_workers=len(self.target_model_types)) as executor:
            # 提交任务
            futures = [executor.submit(self._ask_worker, *t) for t in tasks]
            for f in futures:
                mt, val = f.result()
                sim_dict[mt] = val
        
        # (avg/max)
        return self._summarize(sim_dict, mode)
    
    def _summarize(self, sim_dict, mode='avg'):
        valid_vals = [v for v in sim_dict.values() if v is not None]
        if not valid_vals: return 0, sim_dict
        if mode == 'avg': res = sum(valid_vals) / len(valid_vals)
        elif mode == 'max': res = max(valid_vals)
        return res, sim_dict

    def shutdown(self):
        """关闭所有常驻进程"""
        for model, proc in self.proc_dict.items():
            try:
                proc.stdin.write("EXIT\n")
                proc.stdin.flush()
                proc.terminate()
            except:
                pass

# 服务器
class ShadowMixSimilarityServer:
    def __init__(self, device, env_num, target_model_types, host='127.0.0.1', port=6000):
        self.host = host
        self.port = port
        self.target_model_types = target_model_types
        # 为每个 rank 预先启动好一组常驻进程 (BinCola, CLAP, etc.)
        self.calculators = [ShadowMixSimilarityCalculator(target_model_types, device=device) for rank in range(env_num)]
        # 使用 ThreadPoolExecutor 处理阻塞的管道读取，而不是进程池
        self.thread_executor = ThreadPoolExecutor(max_workers=env_num * 4)
        self.device = self.calculators[0].device

    async def handle_client(self, reader, writer):
        try:
            # 1. 首先读取 4 字节的长度头
            header = await reader.readexactly(4)
            length = struct.unpack('!I', header)[0]

            # 2. 读取指定长度的完整 payload
            data = await reader.readexactly(length)
            
            # 3. 使用 pickle 反序列化（此时 asm1, asm2 已经是 ShadowFunction 对象）
            params = pickle.loads(data)
            
            rank = params.get('rank', 0)
            if rank >= len(self.calculators):
                raise ValueError(f"Invalid rank {rank}, max is {len(self.calculators)-1}")

            calculator = self.calculators[rank]
            loop = asyncio.get_running_loop()

            # 4. 执行计算（此时传入的是对象，calculator 内部可直接调用对象方法）
            sim_result, sim_dict = await loop.run_in_executor(
                self.thread_executor,
                calculator.compare,
                params['asm1'], params['asm2'],
                params['mode']
            )
            
            response = {"status": "ok", "sim_result": sim_result, "sim_dict": sim_dict}
        except Exception as e:
            response = {"status": "error", "message": str(e)}

        # 5. 回复客户端也需要遵循 [长度头 + 数据] 协议，防止客户端读取不全
        resp_data = pickle.dumps(response)
        resp_header = struct.pack('!I', len(resp_data))
        writer.write(resp_header + resp_data)
        
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def run(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(f'[ShadowMixSimilarityServer]: using models {self.target_model_types} start on {self.device}: {self.host}:{self.port}')
        # print(f"[Server] ShadowMixSimilarityServer start on {self.device}: {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

# 客户端类
class ShadowMixSimilarityClient:
    def __init__(self, rank, host='127.0.0.1', port=6000):
        self.host = host
        self.port = port
        self.rank = rank

    async def compare(self, asm1, asm2, mode='avg'):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        
        # 1. 构造整体 Payload（直接放入 ShadowFunction 对象）
        payload = {
            "asm1": asm1, 
            "asm2": asm2,  
            "mode": mode, 
            "rank": self.rank
        }
        
        # 2. 全量序列化
        data_to_send = pickle.dumps(payload)
        
        # 3. 发送长度前缀（防止大对象读取不全）
        # ShadowFunction 包含图结构，可能很大，建议加上 4 字节的长度头
        header = struct.pack('!I', len(data_to_send))
        writer.write(header + data_to_send)
        await writer.drain()
        
        # 4. 读取响应
        # 建议服务器也采用相同的 长度头+Pickle 逻辑
        resp_header = await reader.readexactly(4)
        resp_len = struct.unpack('!I', resp_header)[0]
        resp_data = await reader.readexactly(resp_len)
        
        response = pickle.loads(resp_data)
        
        writer.close()
        await writer.wait_closed()
        
        if response['status'] == 'ok':
            return response['sim_result'], response['sim_dict']
        else:
            raise Exception(response['message'])
import warnings
warnings.filterwarnings("ignore")

import gym
import torch
import numpy as np
import multiprocessing as mp

from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()            
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    # 1. 按照 SB3 惯例，在 info 中保留旧回合的最后一次观测
                    info["terminal_observation"] = observation
                    info["terminal_similarity"] = info["similarity"]
                    info["terminal_similarity_details"] = info["similarity_details"]
                    
                    # 2. 触发重置。因为 SABRE_Wrapper.reset 返回 (obs, info)
                    reset_obs, reset_info = env.reset()
                    
                    # 3. 合并 info 字典
                    info.update(reset_info)
                    
                    # 4. 将观测值替换为新回合的第一帧
                    observation = reset_obs
                
                # 发送后，observation 是 dict，info 也是整合后的 dict
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                # 这里返回的是元组
                result = env.reset()
                if isinstance(result, tuple):
                    observation, info = result
                else:
                    observation, info = result, {}
                remote.send((observation, info))
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break

def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: gym.spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)

def _flatten_infos(infos: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
   return list(infos)

# 并行包装器，将环境包装为支持multiprocess并行的子进程
class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions) -> None:
        # actions 此时是一个 List[Dict]，长度为 n_envs
        for i, remote in enumerate(self.remotes):
            # 直接通过索引取第 i 个环境对应的完整动作字典
            remote.send(("step", actions[i]))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), _flatten_infos(infos)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    # def reset(self) -> VecEnvObs:
    #     for remote in self.remotes:
    #         remote.send(("reset", None))
    #     obs = [remote.recv() for remote in self.remotes]
    #     return _flatten_obs(obs, self.observation_space)

    def reset(self) -> Tuple[VecEnvObs, List[Dict[str, Any]]]:
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        
        obs_list, info_list = zip(*results)
        
        # 使用修正后的 _flatten_infos
        return _flatten_obs(obs_list, self.observation_space), _flatten_infos(info_list)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

# 环境输出的torch包装器，加入这一层包装会让环境输出的obs和reward被包装成tensor
class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    # def reset(self):
    #     obs = self.venv.reset()
    #     for key in obs.keys():
    #         assert isinstance(obs[key], np.ndarray)
    #         obs[key] = torch.from_numpy(obs[key]).to(self.device)
    #     return obs

    def reset(self):
        obs, infos = self.venv.reset() 
        for key in obs.keys():
            obs[key] = torch.from_numpy(obs[key]).to(self.device)
        return obs, infos  # infos 已经是 List[Dict]
    
    def step_async(self, actions):
        """
        actions 输入格式: 包含多个 Tensor 的对象 (B, 1)
        我们需要将其转换为: List[Dict] 长度为 B
        """
        action_to_submit = []
        
        # 1. 自动检测 Batch Size (环境数量)
        # 假设 actions 对象的属性都是 Tensor，取第一个属性的第 0 维
        first_key = list(actions.keys())[0]
        batch_size = actions[first_key].shape[0]

        # 2. 遍历每个环境，拆解动作
        for i in range(batch_size):
            env_action = OrderedDict() # 保持顺序有助于某些环境调试
            for key, value in actions.items():
                if isinstance(value, torch.Tensor):
                    # 提取数值并确保类型正确
                    env_action[key] = value[i].detach().cpu().numpy().item()
            action_to_submit.append(env_action)

        # 3. 将转换后的 List[Dict] 发送给底层的 SubprocVecEnv
        self.venv.step_async(action_to_submit)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        for key in obs.keys():
            obs[key] = torch.from_numpy(obs[key]).to(self.device)

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        # info 原样返回，不再参与任何 flattening
        return obs, reward, done, info

# 支持Dict观测的帧堆叠包装器，专用于以LSTM为核心网络构建的强化学习模型，将环境输出的obs处理为带有历史信息的序列窗口[B, L, obs]
# 在填不满窗口时使用0填充
class VecPyTorchFrameStackDict(VecEnvWrapper):
    """
    专门为 LSTM 设计的 Dict 观测空间帧堆叠包装器。
    产出形状: (Batch, nstack, Feature_Dim...)
    """
    def __init__(self, venv, nstack, device=None, stack_keys=None):
        self.venv = venv
        self.nstack = nstack
        self.device = device if device else torch.device('cpu')
        
        # 指定需要堆叠的键，默认所有键都堆叠
        self.stack_keys = stack_keys if stack_keys else list(venv.observation_space.spaces.keys())
        
        # 1. 创建新的观测空间
        new_spaces = {}
        for key, space in venv.observation_space.spaces.items():
            if key in self.stack_keys and isinstance(space, gym.spaces.Box):
                # 核心修改：在原有 shape 前增加一个维度 [nstack]
                new_shape = (self.nstack,) + space.shape
                
                # 直接获取原始空间的标量边界（取最小值/最大值，或者通常就是 -inf/inf）
                # 传入标量 low 和 high，Gym 会自动将其扩展到 new_shape
                low_val = np.min(space.low) if not np.isscalar(space.low) else space.low
                high_val = np.max(space.high) if not np.isscalar(space.high) else space.high
                
                new_spaces[key] = gym.spaces.Box(
                    low=low_val, 
                    high=high_val, 
                    shape=new_shape, 
                    dtype=space.dtype
                )
            else:
                new_spaces[key] = space
        
        VecEnvWrapper.__init__(self, venv, observation_space=gym.spaces.Dict(new_spaces))
        
        # 2. 初始化堆叠缓冲区
        self.stacked_obs = {}
        for key in self.stack_keys:
            if key in venv.observation_space.spaces:
                space = venv.observation_space.spaces[key]
                if isinstance(space, gym.spaces.Box):
                    # 缓冲区形状: (Batch, Sequence, Feature_Dims...)
                    full_buffer_shape = (venv.num_envs, self.nstack) + space.shape
                    self.stacked_obs[key] = torch.zeros(full_buffer_shape, device=self.device)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        
        for key in self.stack_keys:
            if key in obs and key in self.stacked_obs:
                buffer = self.stacked_obs[key]
                
                # 3. 移动旧帧 (时间轴滑动)
                # 将 1~n 帧向前移动，覆盖 0~n-1 帧
                buffer[:, :-1] = buffer[:, 1:].clone()
                
                # 4. 处理环境重置 (Done)
                for i, done in enumerate(dones):
                    if done:
                        # 如果某个环境结束，将其对应的整个序列缓冲区清零
                        buffer[i].zero_()
                
                # 5. 将最新帧放入序列末尾
                # obs[key] 形状是 (Batch, Feature_Dims...)
                buffer[:, -1] = obs[key]
                obs[key] = buffer
        
        return obs, rews, dones, infos
    
    def reset(self):
        # 统一处理 reset 返回的 (obs, infos)
        obs, infos = self.venv.reset() 
        
        for key in self.stacked_obs:
            # 清空整个缓冲区
            self.stacked_obs[key].zero_()
            
            if key in obs:
                # reset 的初始帧放在序列最后一位
                self.stacked_obs[key][:, -1] = obs[key]
                obs[key] = self.stacked_obs[key]
                
        return obs, infos 
    
    def close(self):
        self.venv.close()

# 环境生产器
class EnvMaker:
    def __init__(self, base_env):
        self.base_env = base_env

    # 构造单个环境的函数
    def make_env(self, env_args, rank):
        def _thunk():
            env = self.base_env(env_args, rank)
            return env

        return _thunk

    # 从主函数调用的多环境构造函数
    def make_vec_envs(self, env_args):
        
        envs = [
            self.make_env(env_args, rank)
            for rank in range(env_args.num_processes)
        ]

        # 多进程环境包装
        if len(envs) > 1:
            envs = SubprocVecEnv(envs)
        # 单进程环境包装
        else:
            envs = DummyVecEnv(envs)        # BUG

        # 对接torch运算的包装器
        envs = VecPyTorch(envs, env_args.sabre_device)

        # 支持LSTM的帧堆叠包装器
        if env_args.num_frame_stack is not None:
            envs = VecPyTorchFrameStackDict(envs, env_args.num_frame_stack, env_args.sabre_device)

        return envs
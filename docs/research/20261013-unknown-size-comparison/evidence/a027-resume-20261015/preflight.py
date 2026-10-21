"""Check the original O2 memory budget on both allocated GPUs without loading weights."""
import inspect
import json
import os
from pathlib import Path

import torch
import transformers
import vllm
from fastfill.system import pipeline

root = Path('/lus/lfs1aip2/projects/u6uv/shanliantian.u6uv/fastfill')
assert Path(pipeline.__file__).resolve().is_relative_to(root / 'snapshots/main-498348274')
engine = pipeline._o2_setup(root / 'models/Qwen3.8-27B')[1]
fraction = engine.get('gpu_memory_utilization',
                      inspect.signature(vllm.LLM.__init__).parameters['gpu_memory_utilization'].default)
assert isinstance(fraction, (float, int)) and 0 < fraction < 1
assert torch.cuda.is_available() and torch.cuda.device_count() == engine['tensor_parallel_size'] == 2
devices = []
for index in range(2):
    free, total = torch.cuda.mem_get_info(index)
    required = fraction * total
    devices.append({'device': index, 'free_gib': free / 1024**3, 'total_gib': total / 1024**3,
                    'required_gib': required / 1024**3, 'ok': free >= required + 512 * 1024**2})
print('GPU_PREFLIGHT', json.dumps({'job': os.environ['SLURM_JOB_ID'], 'devices': devices,
                                 'fraction': fraction, 'torch': torch.__version__, 'vllm': vllm.__version__,
                                 'transformers': transformers.__version__}), flush=True)
assert all(d['ok'] for d in devices), 'Allocated GPU lacks original memory budget plus 0.5 GiB headroom'
print('GPU_PREFLIGHT_OK', flush=True)

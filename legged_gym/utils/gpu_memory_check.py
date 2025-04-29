import torch

def get_tensor_memory_usage(obj, name="root", max_depth=3):
    """
    遍历对象内部所有 tensor，并统计占用显存
    Args:
        obj: 要检查的对象，比如 self、env、policy 等
        name: 起始对象名字
        max_depth: 最大递归深度，防止无限循环
    """
    visited = set()
    var_memory = []
    total_mem = 0

    def recursive_search(obj, prefix, depth):
        nonlocal var_memory, total_mem
        if depth > max_depth:
            return
        if id(obj) in visited:
            return
        visited.add(id(obj))

        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                size_in_bytes = obj.element_size() * obj.numel()
                var_memory.append((prefix, size_in_bytes, tuple(obj.shape)))
                total_mem += size_in_bytes
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                recursive_search(v, f"{prefix}.{k}", depth+1)
        elif isinstance(obj, (list, tuple)):
            for idx, v in enumerate(obj):
                recursive_search(v, f"{prefix}[{idx}]", depth+1)
        else:
            # 普通对象，查属性
            for attr in dir(obj):
                if attr.startswith("__") and attr.endswith("__"):
                    continue
                try:
                    v = getattr(obj, attr)
                    recursive_search(v, f"{prefix}.{attr}", depth+1)
                except Exception:
                    continue

    recursive_search(obj, name, 0)
    var_memory.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== '{name}'对象下的GPU总占用显存：{total_mem/1024/1024:.2f} MB ===\n")
    for n, m, s in var_memory:
        percent = (m / total_mem) * 100 if total_mem > 0 else 0
        print(f"路径: {n:50s} | 显存: {m/1024/1024:.2f} MB | 占比: {percent:.2f}% | shape: {s}")

env = torch.ones([2,2])
get_tensor_memory_usage(env, name="env", max_depth=4)
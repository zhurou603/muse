"""Six nodes, 47 training ranks; physical card 7 on 171 is reserved for A-test."""
NODE_SPECS = [
    ("VM-2-17-tencentos", "172.17.2.17", 8),
    ("VM-2-184-tencentos", "172.17.2.184", 8),
    ("VM-2-46-tencentos", "172.17.2.46", 8),
    ("VM-1-84-tencentos", "172.17.1.84", 8),
    ("VM-0-171-tencentos", "172.17.0.171", 7),
    ("VM-0-178-tencentos", "172.17.0.178", 8),
]
NODES = [(host, ip) for host, ip, _ in NODE_SPECS]
WORLD_SIZE = sum(count for _, _, count in NODE_SPECS)
NOMINAL_GLOBAL_BATCH = WORLD_SIZE * 8


def local_worker_count(host):
    return next(count for name, _, count in NODE_SPECS if name == host)


def rank_location(rank):
    offset = 0
    for host, _, count in NODE_SPECS:
        if offset <= rank < offset + count:
            return host, rank - offset
        offset += count
    raise ValueError(f"Rank {rank} outside 0..{WORLD_SIZE - 1}")

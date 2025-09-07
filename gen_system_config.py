import json
import random
from typing import Optional
from utils import load_config


def load_nodes(config_path: str) -> tuple:
    config = load_config(config_path)
    nodes = [node for node in config["node_config"] if not node.get("disabled")]
    config["train_config"]["worker_num"] = len(nodes)
    return config, nodes


def run_bate_algorithm(nodes: list) -> dict:
    N = len(nodes)
    B = [random.randint(10, 10) for _ in range(N)]
    B.sort(reverse=True)
    
    Bnorm = [0] * N
    graph = {i: [] for i in range(N)}

    def degree(node: int) -> int:
        return sum(1 for u, vs in graph.items() if u == node for _ in vs) + \
               sum(1 for vs in graph.values() for v in vs if v == node)

    Bnorm[0] = B[0]
    for v in range(1, N):
        u = Bnorm.index(max(Bnorm))
        graph[u].append(v)
        Bnorm[u] = B[u] / (degree(u) + 1)
        Bnorm[v] = B[v] / (degree(v) + 1)

    return graph

def build_clients(nodes: list, graph: dict) -> dict:

    def get_node(node_id: int) -> Optional[dict]:
        return next((node for node in nodes if node["id"] == node_id), None)

    clients = {}
    for u, children in graph.items():
        node_u = get_node(u)
        if not node_u:
            continue

        addr_u = node_u["addr"]
        clients.setdefault(addr_u, {
            "id": node_u["id"],
            "mac": node_u["mac"],
            "parent": None,
            "children": []
        })

        for v in children:
            node_v = get_node(v)
            if not node_v:
                continue

            addr_v = node_v["addr"]
            clients[addr_u]["children"].append(node_v)

            if addr_v not in clients:
                clients[addr_v] = {
                    "id": node_v["id"],
                    "mac": node_v["mac"],
                    "parent": node_u,
                    "children": []
                }
            else:
                clients[addr_v]["parent"] = node_u

    return clients


def save_configs(config: dict, clients: dict):
    config["clients"] = clients
    del config["node_config"]

    with open("system_config.json", "w") as f:
        f.write(json.dumps(config, indent=4))

    coord_ip = config["coord_config"]["ip"]
    coord_port = config["coord_config"]["port"]
    env_content = f"COORD_IP={coord_ip}\nCOORD_PORT={coord_port}\n"

    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)

    del config["coord_config"]


def main():
    config, nodes = load_nodes("main_config.json")
    graph = run_bate_algorithm(nodes)
    print("Generated graph:", graph)

    clients = build_clients(nodes, graph)
    save_configs(config, clients)


if __name__ == "__main__":
    main()

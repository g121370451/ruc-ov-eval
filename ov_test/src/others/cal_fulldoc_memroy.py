import json
import numpy as np
from pathlib import Path
import argparse

def count_corpus(corpus_path: Path) -> tuple[int, int]:
    with corpus_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    paragraphs = sum(len(node.get("paragraphs", [])) for node in nodes)
    return len(nodes), paragraphs

def find_embed_path(corpus_path: Path) -> Path:
    stem = corpus_path.stem
    if stem.endswith("_corpus"):
        base = stem[:-len("_corpus")]
    else:
        base = stem
    candidate = corpus_path.parent / f"{base}_emb.npy"
    return candidate if candidate.exists() else None

def estimate_npy_bytes(npy_path: Path) -> tuple[int, int, int]:
    arr = np.load(npy_path, mmap_mode='r')
    rows, cols = arr.shape[0], (arr.shape[1] if len(arr.shape) > 1 else 1)
    size_bytes = rows * cols * 4
    return rows, cols, size_bytes

def human_readable_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def main():
    parser = argparse.ArgumentParser(description="Calculate full document memory usage for DeepRead.")
    parser.add_argument("--index_dir", type=str, required=True, help="Path to the directory containing document indexes.")
    parser.add_argument("--ext", default="_corpus.json", help="corpus file suffix (default: _corpus.json)")
    args = parser.parse_args()

    root = Path(args.index_dir).expanduser().resolve()
    corpus_files = sorted(root.rglob(f"*{args.ext}"))

    from collections import defaultdict
    name_to_paths: dict[str, list[Path]] = defaultdict(list)
    for cp in corpus_files:
        name_to_paths[cp.name].append(cp)
    for name, paths in name_to_paths.items():
        if len(paths) > 1:
            print(f"Warning: Multiple files with name {name} found in {len(paths)} paths:")
            for p in paths:
                print(f"{p.relative_to(root)}")

    if not corpus_files:
        print(f"No corpus files with suffix '{args.ext}' found in {root}")
        return
    
    total_emb_bytes = 0
    total_paragraphs = 0
    total_nodes = 0

    print(f"{'Document':<70} {'Nodes':>7} {'Paragraphs':>7} {'Embeddings':>14} {'EmbMem':>10}")
    print("-" * 110)

    for cp in corpus_files:
        rel = cp.relative_to(root)
        nodes, paras = count_corpus(cp)
        total_nodes += nodes
        total_paragraphs += paras

        emb_path = find_embed_path(cp)
        if emb_path:
            rows, cols, emb_bytes = estimate_npy_bytes(emb_path)
            total_emb_bytes += emb_bytes
            shape_str = f"{rows}x{cols}"
            mem_str = human_readable_size(emb_bytes)
        else:
            shape_str = "N/A"
            mem_str = "N/A"

        name_display = str(rel)
        if len(name_display) > 70:
            name_display = "..." + name_display[-67:]
        print(f"{name_display:<70} {nodes:>7} {paras:>7} {shape_str:>14} {mem_str:>10}")
    
    print("-" * 110)
    print(f"\nSummay")
    print(f"Total documents: {len(corpus_files)}")
    print(f"Total nodes: {total_nodes}")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Total embedding memory: {human_readable_size(total_emb_bytes)}")

if __name__ == "__main__":
    main()
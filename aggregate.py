import argparse
import json
import os

import networkx as nx
from tqdm import tqdm

LANGS = {
    "kirundi",
    "indonesian",
    "ukrainian",
    "spanish",
    "arabic",
    "kyrgyz",
    "thai",
    "azerbaijani",
    "uzbek",
    "igbo",
    "french",
    "serbian_latin",
    "vietnamese",
    "marathi",
    "pidgin",
    "turkish",
    "tigrinya",
    "punjabi",
    "swahili",
    "somali",
    "nepali",
    "hindi",
    "telugu",
    "persian",
    "scottish_gaelic",
    "yoruba",
    "welsh",
    "gujarati",
    "serbian_cyrillic",
    "korean",
    "english",
    "sinhala",
    "tamil",
    "burmese",
    "pashto",
    "amharic",
    "russian",
    "japanese",
    "urdu",
    "portuguese",
    "chinese_simplified",
    "oromo",
    "bengali",
    "hausa",
    "chinese_traditional",
}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-lingual versions of the same document of the CrossSum dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data_path",
        default="./original_data",
        type=str,
        metavar="",
        help="input data directory path",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./aggregated_data",
        type=str,
        metavar="",
        help="output data directory path",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="test",
        type=str,
        metavar="",
        help="split to aggregate (train, val, or test)",
    )
    parser.add_argument(
        "-l",
        "--langs",
        default="all",
        nargs="+",
        type=str,
        metavar="",
        help="languages to download",
    )
    parser.add_argument(
        "--chunk_size",
        default=100,
        type=int,
        metavar="",
        help="number of lines per output file",
    )
    args = parser.parse_args()

    if args.langs == "all":
        langs = LANGS
    else:
        langs = args.langs

    print("Loading data...")
    examples = []
    examples_mono = {}
    for fname in tqdm(os.listdir(os.path.join(args.data_path))):
        if not fname.endswith(f"_{args.split}.jsonl"):
            continue

        source_lang = fname.split("-")[0]
        target_lang = fname.split("-")[1].removesuffix(f"_{args.split}.jsonl")
        if source_lang not in langs or target_lang not in langs:
            continue

        with open(os.path.join(args.data_path, fname), "r") as fd:
            lines = fd.readlines()
        egs = [
            {
                "source_url": json.loads(line)["source_url"],
                "target_url": json.loads(line)["target_url"],
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
            for line in lines
        ]
        examples += egs

        if source_lang == target_lang:
            egs = [json.loads(line) for line in lines]
            examples_mono[source_lang] = {x["source_url"]: x for x in egs}

    print("Building the graph")
    nodes = {}
    graph = nx.Graph()
    dir_edges = {}
    for example in tqdm(examples):
        source_url = example["source_url"]
        target_url = example["target_url"]
        source_lang = example["source_lang"]
        target_lang = example["target_lang"]

        if source_url not in nodes:
            num_nodes = graph.number_of_nodes()
            graph.add_nodes_from(
                [(num_nodes, {"url": source_url, "lang": source_lang})]
            )
            nodes[source_url] = num_nodes

        if source_url not in dir_edges:
            dir_edges[source_url] = set()
        dir_edges[source_url].add(target_url)

        if target_url not in nodes:
            num_nodes = graph.number_of_nodes()
            graph.add_nodes_from(
                [(num_nodes, {"url": target_url, "lang": target_lang})]
            )
            nodes[target_url] = num_nodes

        source_node = nodes[source_url]
        target_node = nodes[target_url]
        graph.add_edge(source_node, target_node)

    print("Finding all maximal cliques")
    cliques = list(nx.find_cliques(graph))

    # discard all cliques with only one document
    cliques = [c for c in cliques if len(c) > 1]

    # discard all cliques that have two documents with the same language
    # as these are very likely to be due to pairing errors in CrossSum
    cliques = [
        c
        for c in cliques
        if len(set([graph.nodes[n]["lang"] for n in c]))
        == len([graph.nodes[n]["lang"] for n in c])
    ]

    # discard all two-document cliques that are not doubly connected
    # as these are very likely to be due to pairing errors in CrossSum
    cliques_filtered = []
    for c in cliques:
        if len(c) > 2:
            cliques_filtered.append(c)
            continue

        source_url = graph.nodes[c[0]]["url"]
        target_url = graph.nodes[c[1]]["url"]
        if target_url in dir_edges[source_url] and source_url in dir_edges[target_url]:
            cliques_filtered.append(c)
    cliques = cliques_filtered

    print("Writing aggregated data to files...")
    num_files, num_lines = 0, 0
    stats = [0] * 4
    fd = None
    for clique in tqdm(cliques):
        line = {}
        examples_clique, langs_clique = [], []
        for node_idx in clique:
            url = graph.nodes[node_idx]["url"]
            lang = graph.nodes[node_idx]["lang"]
            try:
                examples_clique.append(examples_mono[lang][url])
                langs_clique.append(lang)
            except:
                continue

        line["num_docs"] = len(examples_clique)
        for i, (x, lang) in enumerate(zip(examples_clique, langs_clique)):
            line[f"url{i}"] = x["source_url"]
            line[f"lang{i}"] = lang
            line[f"text{i}"] = x["text"]
            line[f"summary{i}"] = x["summary"]

        if line:
            if num_lines == 0:
                fd = open(os.path.join(args.output, f"data_{num_files}.jsonl"), "w")
                num_lines = 0
                num_files += 1
            fd.write(json.dumps(line, ensure_ascii=False) + "\n")
            num_lines += 1
            if num_lines == args.chunk_size:
                fd.close()
                fd = None
                num_lines = 0

        if len(clique) < len(stats):
            stats[len(clique) - 1] += 1
        else:
            stats[-1] += 1
    if fd is not None:
        fd.close()

    print("Aggregation stats:")
    num_cliques = sum(stats)
    print(f"  Found {num_cliques} valid clusters of documents:")
    print(
        "     1 document: {} clusters ({:.1f}%)".format(
            stats[0], 100.0 * stats[0] / num_cliques
        )
    )
    for i in range(1, len(stats) - 1):
        print(
            "     {} documents: {} clusters ({:.1f}%)".format(
                i + 1, stats[i], 100.0 * stats[i] / num_cliques
            )
        )
    print(
        "     {}+ documents: {} clusters ({:.1f}%)".format(
            len(stats), stats[-1], 100.0 * stats[-1] / num_cliques
        )
    )


if __name__ == "__main__":
    cliques = main()

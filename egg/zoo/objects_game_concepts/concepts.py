from typing import List
import xml.etree.ElementTree as et

import argparse
import itertools
import json
import os
import random

random.seed(42)


def load_concepts(skip_homonym=False) -> List[List[str]]:
    """Load concepts from disk. """
    data = {}
    feature_vocab = set()
    category_vocab = set()
    ref_vocab = set()
    for f in os.listdir("resources/concepts/"):
        tree = et.parse("resources/concepts/%s" % f).getroot()
        cat = tree.get("category")
        category_vocab.add(cat)
        for subcat in tree:
            if subcat.tag == "concept":
                subcatname = cat
                concept = subcat
                name = concept.get("name")
                if "_" in name and skip_homonym:
                    continue
                ref_vocab.add(name)
                feats = []
                for aspect in concept:
                    attrs = aspect.text.split()
                    feats += attrs
                    feature_vocab.update(set(attrs))
                data[(cat, subcatname, name)] = feats
            elif subcat.tag == "subcategory":
                subcatname = subcat.get("name")
                category_vocab.add(subcatname)
                for concept in subcat.findall("concept"):
                    name = concept.get("name")
                    if "_" in name and skip_homonym:
                        continue
                    ref_vocab.add(name)
                    feats = []
                    for aspect in concept:
                        attrs = aspect.text.split()
                        feats += attrs
                        feature_vocab.update(set(attrs))
                    data[(cat, subcatname, name)] = feats
            else:
                assert False, "`concept` and `subcategory` should be exhaustive."
    return data, feature_vocab, category_vocab


def save_summary(dataset_path: str, train, test, dev):
    """Summarize the dataset. """
    data = train + test + dev
    features = sorted(list(set(list(itertools.chain.from_iterable(data)))))

    with open(f"{dataset_path}/summary.json", "w") as f:
        json.dump(
            {
                "num_train_items": len(train),
                "num_test_items": len(test),
                "num_dev_items": len(dev),
                "num_feature_sets": len(features),
                "num_feature_values": 1,
                "num_features": len(features),
                "features": features,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_items", type=int, default=400)
    parser.add_argument("--num_test_items", type=int, default=55)
    parser.add_argument("--num_dev_items", type=int, default=55)
    config = parser.parse_args()

    dataset_path = f"./data/concepts"
    if not os.path.exists(f"{dataset_path}"):
        os.mkdir(f"{dataset_path}")

    data, feature_vocab, category_vocab = load_concepts()
    concepts = list(data.values())
    random.shuffle(concepts)
    train = concepts[: config.num_train_items]
    dev = concepts[
        config.num_train_items : config.num_train_items + config.num_dev_items
    ]
    test = concepts[config.num_train_items + config.num_dev_items :]

    with open(f"{dataset_path}/train_1.json", "w") as f:
        json.dump(train, f)
    with open(f"{dataset_path}/test_1.json", "w") as f:
        json.dump(test, f)
    with open(f"{dataset_path}/dev_1.json", "w") as f:
        json.dump(dev, f)

    save_summary(dataset_path, train, test, dev)

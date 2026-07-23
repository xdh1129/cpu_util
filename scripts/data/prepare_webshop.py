#!/usr/bin/env python3
"""Prepare the pinned official WebShop 1k-product environment and Lucene index."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


MIRROR = "zhangdw/webshop"
REVISION = "2fa2bd5dc0cf227e98f6512e71fb88f59eb0c741"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / ".cache/webshop"
    archive = Path(hf_hub_download(MIRROR, "raw/webshop-small.tar.gz", repo_type="dataset", revision=REVISION))
    with tarfile.open(archive) as bundle:
        bundle.extractall(source)
    data = source / "data"
    resources = source / "search_engine/resources_1k"
    resources.mkdir(parents=True, exist_ok=True)
    products = json.loads((data / "items_shuffle_1000.json").read_text())
    with (resources / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for product in products:
            options = ", and ".join(
                f"{name}: {', '.join(values)}"
                for name, values in product.get("options", {}).items()
            )
            contents = " ".join(
                [product["Title"], product["Description"], product["BulletPoints"][0], options]
            ).lower()
            handle.write(json.dumps({"id": product["asin"], "contents": contents, "product": product}) + "\n")
    env = os.environ | {"JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64"}
    subprocess.run(
        [str(root / ".venv-webshop/bin/python"), "-m", "pyserini.index.lucene", "--collection", "JsonCollection", "--input", str(resources), "--index", str(source / "search_engine/indexes_1k"), "--generator", "DefaultLuceneDocumentGenerator", "--threads", "1", "--storePositions", "--storeDocvectors", "--storeRaw"],
        check=True,
        env=env,
    )
    print("Prepared pinned WebShop 1k data and index")


if __name__ == "__main__":
    main()

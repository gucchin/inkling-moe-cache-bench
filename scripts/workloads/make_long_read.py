"""Build long_read.jsonl from documents of the llama.cpp repository (MIT license).

Each prompt asks for a summary of one document. Long documents are cut at a paragraph
boundary so that prompt plus answer stay well inside the 32768 token context.
"""

import json
import os

WORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(WORK, "llama.cpp")
MAX_CHARS = 24000

DOCS = [
    "docs/build.md",
    "docs/function-calling.md",
    "docs/multimodal.md",
    "docs/development/HOWTO-add-model.md",
    "docs/development/token_generation_performance_tips.md",
    "tools/server/README-dev.md",
    "CONTRIBUTING.md",
    "docs/docker.md",
]

ASK = ("Read the document below and summarize it. Give a one-paragraph overview first, "
       "then the five most important points as a list, each with one sentence of explanation.\n\n"
       "--- document: {path} ---\n\n{text}")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_read.jsonl")
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for path in DOCS:
            full = os.path.join(REPO, path)
            if not os.path.isfile(full):
                print("missing:", path)
                continue
            text = open(full, encoding="utf-8").read()
            if len(text) > MAX_CHARS:
                cut = text.rfind("\n\n", 0, MAX_CHARS)
                text = text[: cut if cut > 0 else MAX_CHARS]
            n += 1
            f.write(json.dumps({"id": f"long-{n:02d}", "source": path, "license": "MIT (ggml-org/llama.cpp)",
                                "chars": len(text), "prompt": ASK.format(path=path, text=text)}, ensure_ascii=False) + "\n")
            print(f"long-{n:02d} {path} {len(text)} chars")


if __name__ == "__main__":
    main()

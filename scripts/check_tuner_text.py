"""What text does the tuner's calibration request produce? Same request as moe-cache-tune.py run 1."""

import json
import os

from analyze_trace import tune
from bench import COMMON
from llsrv import WORK, Server, http


def main():
    out = os.path.join(WORK, "runs", "tuner_text")
    srv = Server(COMMON + ["-ngl", "99", "-cmoe", "-ub", "32", "--moe-cache-mib", "7296", "-lm", "none"], os.path.join(out, "server.log"))
    with srv:
        r = http(srv.port, "/completion", {"prompt": tune.PROMPT, "n_predict": 2048, "ignore_eos": True, "cache_prompt": False,
                                           "temperature": 0.8, "top_p": 0.95, "seed": 1})
    text = r["content"]
    open(os.path.join(out, "completion.txt"), "w", encoding="utf-8").write(text)
    words = text.split()
    uniq = len(set(words)) / max(len(words), 1)
    print("chars", len(text), "words", len(words), "distinct word ratio", round(uniq, 3), "t/s", round(r["timings"]["predicted_per_second"], 1))
    for a in (0, len(text) // 2, max(0, len(text) - 300)):
        print("-----", a)
        print(text[a:a + 300].encode("ascii", "backslashreplace").decode())
    json.dump({"chars": len(text), "words": len(words), "distinct_word_ratio": round(uniq, 3)}, open(os.path.join(out, "stats.json"), "w"))


if __name__ == "__main__":
    main()

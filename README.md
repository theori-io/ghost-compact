# Xint's OpenAI Compaction Research

Two major findings in OpenAI's compaction API:

* **[GhostCompact](ghost_compact.md)**: a billing bug that made OpenAI compaction, and the arbitrary inference it can perform, nearly free. Code: [ghost_compact.py](ghost_compact.py), [ghost_scan.py](ghost_scan.py). ([blog post](https://xint.io/blog/openai-compaction-inference))
* **[Compaction Smuggling](compaction_smuggle.md)**: steering compaction to copy a model's hidden reasoning into a compaction item that an older model will read back out. Code: [compaction_smuggle.py](compaction_smuggle.py). ([blog post](https://xint.io/blog/compaction-smuggling-reasoning-traces))

GPT 系列与 BPE 子词切分
GPT vs BERT — 架构对比
BERT (双向编码器)                    GPT (单向解码器)
┌──────────────┐                    ┌──────────────┐
│  Encoder ×12 │  ← 只能"看"        │  Decoder ×12 │  ← 能"生成"
│  双向注意力   │     不生成文本     │  单向注意力   │     逐字生成
└──────────────┘                    └──────────────┘
用途: 理解任务                        用途: 生成任务
  - 分类、NER、问答                     - 对话、写作、翻译

GPT = Generative Pre-trained Transformer，只用了 Transformer 的解码器部分，而且是从左到右单向的，所以天然适合文本生成
为什么需要子词（Subword）切分？
三种切分方式各有问题：
原文: "unbreakable" + "unfriendliness" + "apple"

词级切分（Word-level）:
  → 词典里有就认识，没有就是 [UNK]
  → 遇到 "unbelieveable" 就傻眼了 ❌

字符级切分（Character-level）:
  → "u-n-b-r-e-a-k-a-b-l-e"
  → 每个字符都能认识，但序列太长，计算量大 ❌

子词切分（Subword）:  ← BPE 就是这种！
  → "un" + "break" + "able"
  → 既有词的语义，又能处理新词 ✅

BPE（Byte Pair Encoding）是怎么工作的？
BPE 本质上就是一个不断合并高频字符对的过

语料库: "low" × 5, "lower" × 2, "newest" × 6, "widest" × 3

初始状态:
  l o w <end>
  l o w e r <end>
  n e w e s t <end>
  w i d e s t <end>

语料库: "low" × 5, "lower" × 2, "newest" × 6, "widest" × 3

初始状态:
  l o w <end>
  l o w e r <end>
  n e w e s t <end>
  w i d e s t <end>

第 2 步：继续合并最高频的
es 和 t 频率最高 → 合并为 "est"

  l o w <end>
  l o w e r <end>
  n e w est <end>
  w i d est <end>
第 3 步：继续...
n e  → "ne"
l o  → "lo"
lo w → "low"
最终结果：
"low"     → [low]
"lower"   → [low, er]
"newest"  → [ne, w, est]
"widest"  → [wi, d, est]






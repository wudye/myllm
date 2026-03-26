In jieba's algorithm, working backward is a key part of the dynamic programming approach. Here's how it works:

Backward DP (反向动态规划)
1. Build the DAG First
Construct a Directed Acyclic Graph showing all possible word segments
Each node represents a position in the sentence
2. Calculate from End to Start
Let dp[i] = maximum log probability from position i to the end

dp[n] = 0  (base case: end of sentence)

For i from n-1 down to 0:
    dp[i] = max(dp[i] + logP(word) for all words starting at i)
Where n is the length of the sentence.

3. Why Backward?
When at position i, we can look ahead to see all possible words starting at i
The best choice at i depends on dp[i + len(word)], which is already computed
This avoids recomputation (optimal substructure)
4. Trace Forward
After filling dp[], trace from position 0 following the optimal choices
Reconstruct the best segmentation path
5. Position: 0 1 2 3 4 5 6 7 8 9
Char:     给阿姨倒一杯卡布奇诺

Backward calculation:
dp[9] = 0
dp[8] = logP("诺") + dp[9]
dp[7] = max(logP("奇诺") + dp[9], logP("奇") + dp[8])
dp[6] = max(logP("布奇诺") + dp[9], logP("布") + dp[7])
...
dp[0] = max(logP("给") + dp[1], logP("给阿姨") + dp[2], ...)

Then trace forward: 0 → 2 → 3 → 5 → 9
Result: 给/阿姨/倒/一杯/卡布奇诺
Why This Works
Optimal substructure: Best path from i to end consists of best word at i plus best path from i+len
Memoization: Each position calculated once, stored in dp array
Efficiency: O(n²) vs exponential brute force
This backward DP approach is why jieba can efficiently find the optimal segmentation even for long sentences.
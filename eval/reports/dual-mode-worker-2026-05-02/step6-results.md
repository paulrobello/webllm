# Step 6: Cross-mode token-identical A/B (greedy)

Model: qwen3-0.6b-q4f16
Sampling: temp=0, topK=1, topP=1, rep=1, max=32 tokens
Prompts: 5

| # | Prompt | Match | tokens(main/worker) |
|---|---|:-:|:-:|
| 1 | "What is the capital of France?" | YES | 9/9 |
| 2 | "List the first three prime numbers." | YES | 32/32 |
| 3 | "What is 7 plus 5?" | YES | 10/10 |
| 4 | "Name three primary colors." | YES | 13/13 |
| 5 | "What sound does a dog make?" | YES | 32/32 |

## Verdict: PASS — all byte-identical

## Per-prompt outputs

### Prompt 1: "What is the capital of France?"
```
main:   The capital of France is **Paris**.
worker: The capital of France is **Paris**.
```

### Prompt 2: "List the first three prime numbers."
```
main:   The first three prime numbers are:

1. **2**  
2. **3**  
3. **5**  

These are the smallest prime numbers in
worker: The first three prime numbers are:

1. **2**  
2. **3**  
3. **5**  

These are the smallest prime numbers in
```

### Prompt 3: "What is 7 plus 5?"
```
main:   7 plus 5 equals **12**.
worker: 7 plus 5 equals **12**.
```

### Prompt 4: "Name three primary colors."
```
main:   Three primary colors are **red, blue, and yellow**.
worker: Three primary colors are **red, blue, and yellow**.
```

### Prompt 5: "What sound does a dog make?"
```
main:   A dog makes a variety of sounds depending on its age, breed, and the context in which it is speaking. Some common sounds include:

- **Barks
worker: A dog makes a variety of sounds depending on its age, breed, and the context in which it is speaking. Some common sounds include:

- **Barks
```


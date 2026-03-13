## Dense + Sparse Fusion — The First Retrieval Layer

---

## The Core Problem

When a user asks a question, you need to find the most relevant chunks from potentially thousands stored in ChromaDB. There are two fundamentally different ways to measure "relevant" — and they fail in opposite directions.

```
Question: "What does clause 4.2.1 say about early termination penalties?"

DENSE RETRIEVAL (embeddings) thinks:
  → "early termination penalties" sounds like contract law
  → finds chunks about: cancellation fees, exit clauses, breach penalties
  → MISSES: the exact chunk titled "4.2.1" because semantically it
    looks like every other clause

SPARSE RETRIEVAL (BM25) thinks:
  → "4.2.1" is a rare specific term, score it high
  → finds chunks containing the literal string "4.2.1"
  → MISSES: a chunk that says "early exit fee" (no word overlap)

FUSION: takes both lists, combines them → gets both
```

They fail in opposite directions. Combining them covers both failure modes.

---

## Dense Retrieval — How It Actually Works

```
INGEST TIME (happens once per document):

  "Payment is due within 30 days of invoice date"
                    │
                    ▼
          nomic-embed-text model
          (768-dimensional space)
                    │
                    ▼
  [0.21, -0.43, 0.87, 0.12, ... 768 numbers]
                    │
                    ▼
          stored in ChromaDB on disk
          alongside the original text


QUERY TIME (happens per user question):

  "What are the payment terms?"
                    │
                    ▼
          same nomic-embed-text model
                    │
                    ▼
  [0.19, -0.41, 0.84, 0.15, ... 768 numbers]
                    │
                    ▼
          cosine similarity against ALL stored vectors
          (angle between vectors, not distance)
                    │
                    ▼
          top-K most similar chunks returned
```

**Why cosine similarity works:**

Semantically similar text produces vectors pointing in the same direction in 768-dimensional space. "Payment terms" and "invoice due date" end up near each other even though they share zero words. The model learned this from training on billions of text pairs.

```
                  "invoice due"  →  ●
                                    ↑  small angle = high similarity
"payment terms"  →  ●───────────────┘

"company address" →  ●  (points completely different direction)
                         large angle = low similarity
```

**Where dense retrieval fails:**

```
Question: "What is the contract ID?"
Stored chunk: "Contract ID: AGR-2024-00847"

Embedding of "What is the contract ID?" ≈ generic legal question vector
Embedding of "AGR-2024-00847"           ≈ alphanumeric string vector

These point in very different directions → low cosine score
Dense retrieval MISSES this exact match
```

---

## Sparse Retrieval (BM25) — How It Actually Works

BM25 doesn't use vectors at all. It's pure mathematics on word counts.

**The formula:**

```
BM25 score(chunk, query) =
  Σ  IDF(term) × TF(term, chunk) × (k1 + 1)
 term            ─────────────────────────────────
  in              TF(term, chunk) + k1 × (1 - b + b × len/avglen)
 query

Where:
  TF  = how many times the term appears in this chunk
  IDF = log((N - df + 0.5) / (df + 0.5))   ← how rare the term is
  N   = total number of chunks
  df  = how many chunks contain this term
  k1  = 1.5  (term frequency saturation)
  b   = 0.75 (length normalization factor)
```

Simpler way to think about it:

```
RARE TERM IN CHUNK = HIGH SCORE
  "AGR-2024-00847" appears in 1 out of 500 chunks
  → IDF is high (rare = informative)
  → chunk containing it scores very high for query "AGR-2024-00847"

COMMON TERM IN CHUNK = LOW SCORE
  "the" appears in all 500 chunks
  → IDF is near zero (common = not informative)
  → contributes almost nothing to the score

LENGTH NORMALIZATION:
  A 2000-char chunk containing "payment" once
  scores LOWER than a 100-char chunk containing "payment" once
  → prevents long chunks from winning just by being long
```

**Concrete example:**

```
Query: "clause 4.2.1 termination"
Chunks in DB:

  Chunk A: "...4.2.1 Early Termination. Either party may terminate..."
    "4.2.1"      → appears in 1/500 chunks  → IDF = 3.9  (very rare)
    "termination"→ appears in 12/500 chunks → IDF = 2.7  (rare)
    BM25 score: HIGH ✅

  Chunk B: "...termination of employment requires 30 days notice..."
    "termination"→ appears in 12/500 chunks → IDF = 2.7
    "4.2.1"      → NOT in this chunk        → contributes 0
    BM25 score: MEDIUM

  Chunk C: "...the payment terms are as follows..."
    none of the query terms appear
    BM25 score: 0 ❌
```

**Where BM25 fails:**

```
Query: "What happens if I want to end the agreement early?"

BM25 looks for: "want", "end", "agreement", "early"

Best matching chunk: "4.2.1 Early Termination — either party may
                      cancel this contract before the expiry date"

"want"      → not in chunk  → 0
"end"       → not in chunk  → 0
"agreement" → in chunk      → small score
"early"     → in chunk      → medium score

BM25 score: LOW — misses the most relevant chunk
because the user paraphrased and used different words
```

Dense retrieval would catch this (semantically "end the agreement early" ≈ "early termination").

---

## Reciprocal Rank Fusion — How Fusion Actually Works

Both retrievers return their own ranked lists. RRF merges them.

**The formula:**

```
RRF_score(chunk) = Σ  1 / (rank_i + k)
                  i

  rank_i = position of this chunk in retriever i's list (1-indexed)
  k = 60  (smoothing constant — prevents rank #1 from dominating too much)
  Σ = sum across all retrievers
```

**Step by step example:**

```
Dense retriever returns:          Sparse (BM25) returns:
  Rank 1: Chunk C  (score 0.91)    Rank 1: Chunk A  (score 8.2)
  Rank 2: Chunk A  (score 0.88)    Rank 2: Chunk D  (score 6.1)
  Rank 3: Chunk B  (score 0.81)    Rank 3: Chunk C  (score 4.8)
  Rank 4: Chunk D  (score 0.74)    Rank 4: Chunk B  (score 3.2)
  Rank 5: Chunk E  (score 0.61)    Rank 5: Chunk F  (score 2.1)

RRF calculation (k=60):

  Chunk A: 1/(2+60) + 1/(1+60) = 0.01613 + 0.01639 = 0.03252  ← WINNER
  Chunk C: 1/(1+60) + 1/(3+60) = 0.01639 + 0.01587 = 0.03226
  Chunk D: 1/(4+60) + 1/(2+60) = 0.01563 + 0.01613 = 0.03176
  Chunk B: 1/(3+60) + 1/(4+60) = 0.01587 + 0.01563 = 0.03150
  Chunk E: 1/(5+60) + 0         = 0.01538 + 0       = 0.01538
  Chunk F: 0         + 1/(5+60) = 0       + 0.01538 = 0.01538

Final fused ranking:
  1. Chunk A  (0.03252) ← appeared #2 dense AND #1 sparse
  2. Chunk C  (0.03226) ← appeared #1 dense AND #3 sparse
  3. Chunk D  (0.03176) ← appeared #4 dense AND #2 sparse
  4. Chunk B  (0.03150)
  5. Chunk E  (0.01538) ← only in dense
  5. Chunk F  (0.01538) ← only in sparse
```

**The key insight:** Chunk A wins not because it ranked #1 in either retriever, but because it ranked highly in **both**. RRF rewards agreement between retrieval methods. A chunk that both methods think is relevant is almost certainly actually relevant.

---

## Your DocuMind Code — Annotated

```python
def build_retriever(doc_ids=None, use_reranking=True):

    # ── Layer 1a: Dense ──────────────────────────────────────
    # nomic-embed-text embeds the question at query time
    # ChromaDB finds top-10 nearest vectors (cosine similarity)
    dense_retriever = _build_dense_retriever(chroma_filter)
    # returns: [ChunkC(0.91), ChunkA(0.88), ChunkB(0.81), ...]

    # ── Layer 1b: Sparse ─────────────────────────────────────
    # Fetches ALL chunk texts from ChromaDB
    # Builds BM25 index in memory (rebuilt each query)
    # Scores chunks by term frequency + rarity
    sparse_retriever = _build_sparse_retriever(doc_ids)
    # returns: [ChunkA(8.2), ChunkD(6.1), ChunkC(4.8), ...]

    # ── Layer 1 fusion: RRF ──────────────────────────────────
    base_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4],
        # weights scale the RRF scores before combining:
        # dense_score × 0.6 + sparse_score × 0.4
        # gives dense slightly more influence
    )
    # returns: [ChunkA(0.0325), ChunkC(0.0323), ChunkD(0.0318), ...]

    # ── Layer 2: FlashRank re-ranks the top-10 to top-3 ──────
    return _wrap_with_flashrank(base_retriever)
```

---

## When to Tune the Weights

Your current weights `[0.6, 0.4]` are a solid default. Here's when to change them:

```
Document type             Recommended weights     Why
─────────────────────────────────────────────────────────────────
General docs (default)    [0.6, 0.4]              balanced
Legal contracts           [0.4, 0.6]              clause numbers, exact terms
Technical documentation   [0.4, 0.6]              function names, error codes
Research papers / books   [0.7, 0.3]              conceptual, paraphrased
Customer support FAQs     [0.65, 0.35]            mix of semantic + keywords
Code repositories         [0.3, 0.7]              exact symbol/method names
```

Change one line in `_build_ensemble_retriever()`:

```python
return EnsembleRetriever(
    retrievers=[dense, sparse],
    weights=[0.4, 0.6],   # ← tune this
)
```

---

## Complete Mental Model

```
                        USER QUESTION
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    DENSE (nomic-embed-text)         SPARSE (BM25)
    "what does this MEAN?"           "what WORDS are here?"
    finds paraphrases                finds exact terms
    finds synonyms                   finds IDs, codes, names
    misses exact matches             misses paraphrases
              │                             │
              │  top-10 chunks              │  top-10 chunks
              │  (different lists)          │  (different lists)
              └──────────────┬──────────────┘
                             │
                             ▼
                    RRF FUSION [0.6, 0.4]
                    "which chunks did BOTH agree on?"
                    chunks in both lists score highest
                             │
                             │  top-10 fused chunks
                             ▼
                    FLASHRANK (layer 2)
                    cross-encoder reads Q+chunk together
                             │
                             │  top-3 precise chunks
                             ▼
                    LLM PROMPT
                    llama3.2 answers from clean context
```

The reason this matters: each layer catches what the previous layer misses. Dense catches semantics, sparse catches keywords, RRF catches agreement, FlashRank catches the remaining noise. By the time 3 chunks reach `llama3.2`, they are the 3 best chunks in your entire document collection for that specific question.

# Word Embeddings → Transformers → Attention Mechanism

### From First Principles to Full GPT Architecture

---

# 📋 TABLE OF CONTENTS

```
1.  Intro — Why This Topic Matters
2.  Representing Images into Numbers
3.  Representing Text into Numbers
4.  One-Hot Encoding
5.  Bag of Words (Unigram, Bigram, N-Gram)
6.  Semantic & Contextual Understanding
7.  Word Embeddings
8.  Visualizing Word2Vec Embeddings (with PCA)
9.  Word2Vec Training — CBOW & Skip-Gram + Sliding Window
10. Embedding Layer in Transformer Architecture
11. Positional Encoding
12. Attention Mechanism
13. Full GPT Architecture (Complete Pipeline)
14. Master Summary Sheet
```

---

# 1. 🎯 INTRO — WHY THIS TOPIC MATTERS

Before any machine learning model can process language, it faces one fundamental problem:

```
Computers only understand numbers.
Language is made of symbols, words, and meaning.
```

The entire field of NLP (Natural Language Processing) is built around answering one question:

> **How do we convert words into numbers without destroying their meaning?**

This is not trivial. Consider:

- "King" and "Queen" are related — any good representation must reflect this
- "Bank" means different things in different sentences — any good representation must capture this
- "The cat sat" and "Sat the cat" mean different things — order matters

The journey from raw text → numbers → meaning is exactly what these notes cover.

---

# 2. 🖼️ REPRESENTING IMAGES INTO NUMBERS

Before text, let's understand how images are handled — it gives intuition for why text is harder.

### Images are naturally numerical:

```
Grayscale image (28×28 pixels):

[ 0,   0,   0,  255, 255,  0,   0  ]
[ 0,   0,  255, 255, 255, 255,  0  ]
[ 0,  255, 255,  0,   0,  255, 255 ]
...

Each pixel = one number (0 to 255)
Full image = 28 × 28 = 784 numbers
```

For **RGB (colour) images:**

```
Each pixel has 3 channels: Red, Green, Blue
Image size: (Height × Width × 3)

Example: 224×224 colour image
= 224 × 224 × 3 = 150,528 numbers
```

### Why images are easier than text:

```
Image pixel values:    0 → 255  (fixed, bounded, continuous)
Text word values:      "king", "democracy", "photosynthesis"...
                       → unbounded, symbolic, context-dependent
```

> 💡 **Insight:** A pixel at position (3,4) with value 128 is always 128. But the word "light" in "light bulb" vs "light rain" carries completely different meaning. Images are spatially structured. Text is semantically structured. That's what makes NLP hard.

---

# 3. 📝 REPRESENTING TEXT INTO NUMBERS

The first naive approach: just assign each word a unique integer.

```
Vocabulary:
"the"   → 1
"cat"   → 2
"sat"   → 3
"on"    → 4
"mat"   → 5

Sentence: "the cat sat on the mat"
→ [1, 2, 3, 4, 1, 5]
```

### Why this fails immediately:

```
"cat" = 2
"mat" = 5

Is mat 2.5× more important than cat? NO.
Does the number 5 mean anything about "mat"? NO.
Is "sat" (3) halfway between "cat" (2) and "on" (4)? NO.
```

Arbitrary integers imply **false mathematical relationships.** The model will think words with higher numbers are "greater" or "more important" — which is meaningless.

We need a better approach. This leads to the methods below.

---

# 4. ❌ ONE-HOT ENCODING (Timestamp: 2:20)

### The Idea:

Give every word its own dedicated dimension. Put a `1` in that word's position and `0` everywhere else.

```
Vocabulary: [king, queen, man, woman, royal, commoner]
              idx:   0     1    2      3      4        5

"king"   → [1, 0, 0, 0, 0, 0]
"queen"  → [0, 1, 0, 0, 0, 0]
"man"    → [0, 0, 1, 0, 0, 0]
"woman"  → [0, 0, 0, 1, 0, 0]
"royal"  → [0, 0, 0, 0, 1, 0]
```

Each vector has length = vocabulary size. Exactly one `1`, rest are `0`.

### Mathematical Properties:

**Dot product between any two one-hot vectors:**

```
king · queen = [1,0,0,0,0,0] · [0,1,0,0,0,0]
             = (1×0) + (0×1) + (0×0) + ...
             = 0

king · king  = [1,0,0,0,0,0] · [1,0,0,0,0,0]
             = 1
```

**Cosine similarity between any two different words:**

```
cos(king, queen) = 0 / (1 × 1) = 0
cos(king, banana) = 0 / (1 × 1) = 0
```

**King and queen are as similar as king and banana — ZERO in both cases.**

### In Python:

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

vocabulary = [["king"], ["queen"], ["man"], ["woman"]]
encoder = OneHotEncoder(sparse=False)
one_hot = encoder.fit_transform(vocabulary)

print(one_hot)
# [[1. 0. 0. 0.]   ← king
#  [0. 1. 0. 0.]   ← queen
#  [0. 0. 1. 0.]   ← man
#  [0. 0. 0. 1.]]  ← woman

# With real vocab of 50,000 words:
# Each vector = 50,000 numbers
# 49,999 of them are ZERO
```

### Problems with One-Hot Encoding:

```
Problem 1 — SPARSITY
Vocabulary = 50,000 words
Each vector = 50,000 numbers
99.998% of numbers = 0
This is called a SPARSE vector — mostly empty, huge waste of memory

Problem 2 — NO SEMANTIC MEANING
king ↔ queen distance = same as king ↔ banana distance
The vectors know nothing about meaning

Problem 3 — NO RELATIONSHIP
"good" and "great" are synonyms → but their vectors are orthogonal (90°)
The encoding has completely missed this connection

Problem 4 — FIXED SIZE
Add one new word → must rebuild ALL vectors from scratch
```

> 💡 **Insight:** One-hot encoding is like giving every person in a city their own radio frequency — you can identify them individually, but the frequencies don't tell you anything about who they are or how they relate to each other.

---

# 5. ❌ BAG OF WORDS — Unigram, Bigram, N-Gram (Timestamp: 3:40)

### The Idea:

Instead of encoding individual words, encode **entire sentences/documents** by counting how many times each word appears. Ignore word order completely.

```
Vocabulary: [the, cat, sat, on, mat, dog, ran]

Sentence 1: "the cat sat on the mat"
→ {the:2, cat:1, sat:1, on:1, mat:1, dog:0, ran:0}
→ [2, 1, 1, 1, 1, 0, 0]

Sentence 2: "the mat sat on the cat"
→ {the:2, mat:1, sat:1, on:1, cat:1, dog:0, ran:0}
→ [2, 1, 1, 1, 1, 0, 0]   ← IDENTICAL VECTOR!
```

Both sentences produce the **exact same vector** even though their meaning is different.

---

### Unigram, Bigram, N-Gram:

**Unigram** = each single word is a token

```
"the cat sat" → ["the", "cat", "sat"]
```

**Bigram** = pairs of consecutive words are tokens

```
"the cat sat" → ["the cat", "cat sat"]
```

**N-Gram** = sequences of N consecutive words

```
N=3: "the cat sat" → ["the cat sat"]
N=2: "the quick brown fox" → ["the quick", "quick brown", "brown fox"]
```

Bigrams and N-grams capture **some local context** — "New York" as a bigram is more informative than "New" and "York" separately.

### TF-IDF (Improved BoW):

Rather than raw counts, weight words by how **unique** they are to a document:

```
TF  (Term Frequency)     = count of word in THIS document
                           ─────────────────────────────
                           total words in THIS document

IDF (Inverse Doc Freq)   = log(total documents / documents containing word)

TF-IDF = TF × IDF
```

Words like "the", "a", "is" appear everywhere → low IDF → low score
Words like "photosynthesis" are rare → high IDF → high score

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "the cat sat on the mat",
    "the dog ran on the road",
    "the cat and the dog are friends"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### Problems with Bag of Words:

```
Problem 1 — ORDER DESTROYED
"dog bites man" = "man bites dog" → SAME vector
Meaning is completely lost

Problem 2 — STILL SPARSE
Even with bigrams, matrix is huge and mostly zeros

Problem 3 — NO SEMANTIC UNDERSTANDING
"good", "great", "excellent" are unrelated in BoW
Synonyms look as different as opposites

Problem 4 — CONTEXT BLIND
"I love this movie" (positive)
"I don't love this movie" (negative)
BoW gives these very similar vectors — misses "don't"
```

> 💡 **Insight:** BoW is like judging a book by counting its words. You know "war" appears 200 times and "peace" appears 150 times — but you have no idea if it's a war novel or a peace manifesto. Counting without ordering is like music without rhythm.

---

# 6. 🧠 SEMANTIC & CONTEXTUAL UNDERSTANDING (Timestamp: 4:59)

### What Semantic Understanding Means:

**Semantic** = relating to meaning

A good text representation should capture:

```
SYNONYMY:     good ≈ great ≈ excellent  (similar meaning)
ANTONYMY:     hot ↔ cold               (opposite meaning)
ANALOGY:      king:queen = man:woman   (relational meaning)
HYPERNYMY:    dog → animal → living thing (hierarchical)
POLYSEMY:     bank = river bank OR financial bank (multiple meanings)
```

### The Distributional Hypothesis:

The foundational theory behind modern embeddings, by linguist J.R. Firth (1957):

> **"You shall know a word by the company it keeps."**

Words that appear in **similar contexts** tend to have **similar meanings.**

```
"I drink ___"   → water, coffee, juice, tea, wine
"I eat  ___"   → food, pizza, rice, bread, cake

"coffee" and "tea" appear after "drink" frequently
→ they should have SIMILAR vectors
```

This is the insight that makes Word2Vec work.

### What Contextual Understanding Means:

**Contextual** = meaning changes depending on surrounding words

```
"I sat on the bank of the river"  →  bank = land beside water
"I deposited money at the bank"   →  bank = financial institution
"I bank on you to help me"        →  bank = rely on
```

The same word "bank" needs **three different vectors** depending on context.

One-Hot and BoW give it **one fixed vector** for all cases → fail.
Word2Vec gives it **one fixed vector** → partial improvement, still fails.
Transformers give it **context-dependent vectors** → succeeds completely.

---

# 7. ✅ WORD EMBEDDINGS (Timestamp: 6:28)

### The Core Idea:

Instead of sparse, high-dimensional vectors (One-Hot) — represent each word as a **small, dense vector** of real numbers in a **continuous vector space.**

```
One-Hot:  "king" → [0,0,0,1,0,0,0,...,0]   ← 50,000 numbers, 49,999 zeros
Embedding: "king" → [0.91, 0.23, 0.76, 0.44, ...]  ← just 300 numbers, all meaningful
```

### The Embedding Space:

```
            Royalty axis
                 ↑
    queen •      |      • king
                 |
─────────────────────────────── Gender axis
                 |
   woman •       |      • man
                 ↓
```

Words with **similar meanings** cluster **closer together** in this space.

The distance between vectors measures semantic similarity:

```
Euclidean distance:
d(king, queen) = √[(0.91-0.89)² + (0.23-0.21)² + ...]  ← SMALL

d(king, banana) = √[(0.91-0.02)² + (0.23-0.88)² + ...] ← LARGE
```

Cosine similarity (more commonly used):

```
         A · B
cos(A,B) = ─────
           |A||B|

cos(king, queen) ≈ 0.89   ← very similar, close to 1
cos(king, banana) ≈ 0.12  ← very different, close to 0
```

### 🧮 The Famous Analogy — King − Man + Woman = Queen:

```
vector("king")   = [0.91, 0.23, 0.76, ...]
vector("man")    = [0.45, 0.11, 0.24, ...]
vector("woman")  = [0.43, 0.88, 0.25, ...]
vector("queen")  = [0.89, 0.90, 0.77, ...]

king - man + woman
= [0.91-0.45+0.43, 0.23-0.11+0.88, 0.76-0.24+0.25, ...]
= [0.89,           1.00,            0.77,            ...]
≈ vector("queen") ✅
```

**Why this works geometrically:**

```
The vector from "man" to "king" = direction of "royalty"
The vector from "woman" to "queen" = SAME direction of "royalty"

king - man  =  royalty direction
woman + royalty direction  =  queen
```

> 💡 **Insight:** This moment is when NLP became magical. Nobody told the model what "royalty" or "gender" means. It discovered these abstract concepts purely from reading patterns in text. Meaning is geometry.

```python
import numpy as np

# Simplified 3D embeddings (real ones are 300-1000D)
vectors = {
    "king":  np.array([0.91, 0.23, 0.76]),
    "queen": np.array([0.89, 0.90, 0.77]),
    "man":   np.array([0.45, 0.11, 0.24]),
    "woman": np.array([0.43, 0.88, 0.25]),
}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# king - man + woman
result = vectors["king"] - vectors["man"] + vectors["woman"]

# Find closest word
for word, vec in vectors.items():
    print(f"{word}: {cosine_similarity(result, vec):.4f}")
# queen should score highest
```

---

# 8. 📊 VISUALIZING WORD2VEC EMBEDDINGS WITH PCA (Timestamp: 9:44)

### The Problem with High Dimensions:

Word2Vec produces vectors of 300 dimensions. You cannot plot 300-dimensional space. Humans can only see 2D or 3D.

**PCA (Principal Component Analysis)** solves this — it compresses high-dimensional vectors down to 2D while **preserving the most important structure.**

### How PCA Works (Intuition):

```
300-dimensional embedding space:
    Many dimensions carry little information
    A few dimensions carry most information

PCA finds the 2 directions that capture the MOST VARIANCE
(the 2 directions where words spread out the most)
and projects everything onto those 2 directions
```

### The Mathematics of PCA:

```
Step 1: Centre the data
    X_centered = X - mean(X)

Step 2: Compute covariance matrix
    C = (1/n) × X_centeredᵀ × X_centered
    shape: (300 × 300)

Step 3: Compute eigenvectors and eigenvalues of C
    C × v = λ × v
    v = eigenvector (a direction)
    λ = eigenvalue  (how much variance in that direction)

Step 4: Sort eigenvectors by eigenvalue (highest first)
    These are your "principal components"

Step 5: Project data onto top 2 components
    X_2d = X_centered × [v1, v2]
    shape: (n_words × 2) ← now plotable!
```

### In Python:

```python
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Sample training data
sentences = [
    ["king", "rules", "the", "kingdom"],
    ["queen", "rules", "the", "kingdom"],
    ["man", "and", "woman", "are", "humans"],
    ["paris", "is", "the", "capital", "of", "france"],
    ["berlin", "is", "the", "capital", "of", "germany"],
    ["dog", "and", "cat", "are", "animals"],
]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=100, window=3,
                 min_count=1, sg=1, epochs=200)

# Words to visualize
words = ["king", "queen", "man", "woman",
         "paris", "berlin", "france", "germany",
         "dog", "cat"]

# Get their vectors
vectors = np.array([model.wv[w] for w in words])  # (10, 100)

# PCA: compress 100D → 2D
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)  # (10, 2)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))
plt.title("Word2Vec Embeddings Visualized with PCA")
plt.show()
```

### What You See When You Plot:

```
                    |
  queen •   king •  |
                    |
  woman •   man  •  |          paris • berlin •
─────────────────────────────────────────────────
                    |      france • germany •
                    |
          dog • cat •
                    |
```

- **Royalty words cluster together** (king, queen)
- **Gender words cluster together** (man, woman)
- **Cities cluster together** (paris, berlin)
- **Countries cluster separately** (france, germany)
- **Animals cluster together** (dog, cat)

**The model learned all of this from raw text — with zero human-defined categories.**

> 💡 **Insight:** PCA is the X-ray machine for embeddings. It lets you see the invisible geometry that the model has built inside 300 dimensions. When you see words clustering by category, you know the embeddings have genuinely captured semantic meaning.

---

# 9. 🏋️ WORD2VEC TRAINING — CBOW & SKIP-GRAM + SLIDING WINDOW (Timestamp: 10:30)

### The Key Insight Behind Word2Vec:

Word2Vec doesn't directly "learn embeddings." Instead it trains a neural network to **predict words from context.** The embeddings are a **byproduct** of this prediction task.

```
Fake task:   predict a word from its neighbours
Real prize:  the weight matrix = word embeddings
```

---

## 9A. THE SLIDING WINDOW

### From the image you shared:

```
"The quick brown fox jumps over the lazy dog"
                   ↑      ↑      ↑
                  fox   jumps   over
              (Context) (Target) (Context)

Window size = 3, Context size = 1
```

**The window slides one word at a time across the entire sentence:**

```
Position 1:  [The]  →  context: {},  target: The
Position 2:  [The quick]  →  context: {The},  target: quick
Position 3:  [The quick brown]  →  context: {The, brown},  target: quick
Position 4:  [quick brown fox]  →  context: {quick, fox},  target: brown
Position 5:  [brown fox jumps]  →  context: {brown, jumps}, target: fox
Position 6:  [fox jumps over]   →  context: {fox, over},   target: jumps
Position 7:  [jumps over the]   →  context: {jumps, the},  target: over
...
```

Each window position generates **training pairs** that teach the model which words appear together.

### Effect of Window Size:

```
Small window (1-2):   captures SYNTACTIC relationships
                      "quick" and "brown" → both adjectives
                      model learns grammar

Large window (5-10):  captures SEMANTIC relationships
                      "jumps" and "dog" → same sentence topic
                      model learns meaning
```

```python
sentence = ["The","quick","brown","fox","jumps","over","the","lazy","dog"]
window_size = 1

def generate_pairs(sentence, window_size):
    pairs = []
    for i, target in enumerate(sentence):
        for j in range(i - window_size, i + window_size + 1):
            if j != i and 0 <= j < len(sentence):
                pairs.append((target, sentence[j]))
    return pairs

pairs = generate_pairs(sentence, window_size=1)
for p in pairs:
    print(f"  Target: {p[0]:8} | Context: {p[1]}")
```

---

## 9B. CBOW — Continuous Bag of Words

**Direction: Context → Predict Target**

```
Input:  context words   [fox, over]
Output: predict target  [jumps]
```

### Neural Network Architecture:

```
Input Layer:
    context word vectors (averaged or summed)
    shape: (1 × embedding_dim)

    [fox vector] + [over vector]
    ─────────────────────────── = average context vector
                2

Hidden Layer (Projection):
    Linear transformation
    shape: (embedding_dim × embedding_dim)

Output Layer:
    Softmax over entire vocabulary
    shape: (1 × vocab_size)
    → probability distribution over all words
    → highest probability should be "jumps"
```

### Mathematical objective (maximize):

```
CBOW objective = maximize:

log P(wt | wt-c, ..., wt-1, wt+1, ..., wt+c)

Where:
    wt   = target word
    wt-c to wt+c = context words within window c
```

CBOW is **faster** and works better for **frequent words.**

---

## 9C. SKIP-GRAM

**Direction: Target → Predict Context**

```
Input:  center/target word   [jumps]
Output: predict context       [fox, over]
```

### Neural Network Architecture:

```
Input Layer:
    center word one-hot vector
    shape: (1 × vocab_size)

Hidden Layer (Embedding):
    shape: (vocab_size × embedding_dim)
    ← THIS IS THE EMBEDDING MATRIX WE WANT

Output Layer:
    Softmax over vocabulary
    shape: (1 × vocab_size)
    → one probability distribution per context word
```

### Mathematical objective (maximize):

```
Skip-Gram objective = maximize:

(1/T) Σ Σ log P(wt+j | wt)
       t  -c≤j≤c, j≠0

Where:
    T = total words in corpus
    c = window size
    wt = center word
    wt+j = context words
```

Skip-Gram works better for **rare words** and gives **higher quality embeddings** — at the cost of being slower.

---

## 9D. THE ACTUAL NEURAL NETWORK

```
Input: one-hot vector of center word  →  shape: (V,)  V = vocab size

         ┌─────────────────────────┐
         │   Hidden Layer          │
         │   W_in: (V × D)         │  ← THIS becomes the embedding matrix
         │   output: (D,)          │  D = embedding dimension (e.g. 300)
         └─────────────────────────┘
                    ↓
         ┌─────────────────────────┐
         │   Output Layer          │
         │   W_out: (D × V)        │
         │   output: (V,)          │
         └─────────────────────────┘
                    ↓
              Softmax → probabilities
                    ↓
         Cross-entropy loss with true context words
                    ↓
              Backpropagation
                    ↓
         Update W_in and W_out
```

After training on millions of sentences:

- **W_in** = the embedding matrix (row i = embedding for word i)

---

## 9E. TRAINING TRICKS FROM THE ORIGINAL PAPER

### Trick 1 — Hierarchical Softmax:

Standard softmax over 50,000 words is **expensive.** Hierarchical softmax organizes vocabulary into a **binary tree:**

```
                    root
                  /      \
              /               \
         ...                    ...
        /   \                  /   \
     king  queen           man    woman
```

To compute probability of "king":

- Navigate from root to "king" leaf
- Multiply probabilities at each binary decision
- Complexity: O(log V) instead of O(V)

### Trick 2 — Negative Sampling:

Instead of updating weights for **all 50,000 words**, only update:

- The **1 correct** context word (positive sample)
- **5-20 random wrong** words (negative samples)

```python
# Negative sampling objective
# Maximize: log σ(v_wo · v_wi) + Σ log σ(-v_wk · v_wi)
# Where:
#   v_wo = output vector of correct word
#   v_wi = input vector of center word
#   v_wk = output vectors of k negative (random) words
#   σ    = sigmoid function
```

This makes training **10-100× faster** with minimal quality loss.

### Trick 3 — Subsampling of Frequent Words:

Words like "the", "a", "in" appear millions of times but carry little meaning. They are randomly **discarded during training** with probability:

```
P(discard word w) = 1 - √(t / f(w))

Where:
    f(w) = frequency of word w in corpus
    t    = threshold (typically 10⁻⁵)
```

Words appearing very frequently get discarded most often → the model focuses on meaningful words.

---

## 9F. Full Word2Vec in Python (Gensim):

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "the", "kingdom"],
    ["man", "and", "woman", "are", "people"],
    ["paris", "is", "capital", "of", "france"],
]

model = Word2Vec(
    sentences,
    vector_size=100,    # embedding dimensions
    window=5,           # context window size
    min_count=1,        # ignore words appearing less than this
    sg=1,               # 0 = CBOW, 1 = Skip-Gram
    negative=5,         # negative samples
    epochs=100
)

# Get vector
print(model.wv["king"])          # 100-dim vector

# Similarity
print(model.wv.similarity("king", "queen"))  # e.g. 0.89

# Analogy: king - man + woman = ?
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=3
)
print(result)  # [("queen", 0.89), ...]
```

---

# 10. 🤖 EMBEDDING LAYER IN TRANSFORMER ARCHITECTURE (Timestamp: 14:46)

### From the image you shared (Embedding → Decoder → Unembedding):

```
Input text (One Hot Encoded Token)
              ↓
         EMBEDDING
    (compress: vocab_size → d_model)
              ↓
    Low Dimensional Representation
         (Embedding) [dense]
              ↓
      Decoder Block × N
              ↓
         UNEMBEDDING
    (expand: d_model → vocab_size)
              ↓
    Output Probabilities (Next Token)
    [cat, at, tokyo, #ization, ?, they, nope...]
```

---

## 10A. THE EMBEDDING LAYER

### How It Works:

The embedding layer is a **lookup table** — a matrix of shape `(vocab_size × d_model)`.

```
vocab_size = 50,000    (total unique tokens)
d_model    = 512       (embedding dimension)

Embedding matrix E: shape (50,000 × 512)
                           ↑         ↑
                     50,000 rows   512 columns
                     one per word  one per dimension
```

**Input:** One-hot vector of shape `(50,000,)`

**Operation:** Matrix multiplication

```
one_hot_vector (50,000,) × E (50,000 × 512) = embedding (512,)
```

**Efficient trick:** Since one-hot has only one `1`, this is just a **row lookup:**

```
token_id = 3 (for "fox")
embedding = E[3, :]    ← just grab row 3 from the matrix
```

```python
import torch
import torch.nn as nn

vocab_size = 50000
d_model = 512

# The ENTIRE left network in the architecture image
embedding_layer = nn.Embedding(vocab_size, d_model)

# Sentence: "The cat sat" → token IDs [464, 3797, 3332]
token_ids = torch.tensor([464, 3797, 3332])

# Get embeddings
embeddings = embedding_layer(token_ids)  # shape: (3, 512)

print(embeddings.shape)  # torch.Size([3, 512])
```

---

## 10B. THE UNEMBEDDING LAYER

The **exact reverse** of embedding — expands from `d_model` back to `vocab_size`:

```
Decoder output: (seq_len × 512)
      ↓
Unembedding (Linear layer): W_u shape (512 × 50,000)
      ↓
Logits: (seq_len × 50,000) ← one score per word
      ↓
Softmax: (seq_len × 50,000) ← probabilities
      ↓
Next token = argmax(probabilities[-1, :])
```

```python
# Unembedding = linear layer (no bias, maps back to vocab)
unembedding = nn.Linear(512, 50000, bias=False)

decoder_output = torch.randn(1, 3, 512)    # (batch, seq_len, d_model)
logits = unembedding(decoder_output)       # (1, 3, 50000)
probs  = torch.softmax(logits, dim=-1)     # (1, 3, 50000)

# Get next token
next_token_id = torch.argmax(probs[:, -1, :], dim=-1)
```

---

## 10C. WEIGHT TYING

In GPT-2 and many modern models, the **embedding and unembedding share the same weight matrix** (transposed):

```
Embedding matrix:    E  shape (50,000 × 512)
Unembedding matrix:  Eᵀ shape (512 × 50,000)  ← just transpose!
```

```python
# Weight tying
model.unembedding.weight = model.embedding.weight
```

**Why?** Forces consistency — the representation of a word going **in** matches the representation coming **out.** Also halves the number of parameters.

---

## 10D. DIFFERENCE FROM WORD2VEC EMBEDDINGS:

```
Word2Vec Embeddings:
    → Pretrained on large corpus
    → Fixed after training (frozen)
    → Same vector regardless of context
    → "bank" always = [0.23, 0.44, ...]

Transformer Embedding Layer:
    → Initialized randomly
    → Updated during task-specific training via backpropagation
    → Learns representations specifically for the task
    → Contextual (after attention): "bank" = different vectors in different sentences
```

---

# 11. 📍 POSITIONAL ENCODING (Timestamp: 17:16)

### The Problem:

Transformers process **all tokens simultaneously** (in parallel). Unlike RNNs that read left-to-right, Transformers have no built-in concept of order.

```
"The cat sat on the mat"
"The mat sat on the cat"

Both produce identical embedding matrices if order is ignored!
```

Without position information, the model cannot distinguish these.

### The Solution — Add Position Vectors:

```
Final Input = Word Embedding + Positional Encoding
              (seq_len × 512) + (seq_len × 512)
            = (seq_len × 512)
```

---

### Method 1 — Fixed Sinusoidal (Original Paper: Vaswani et al. 2017):

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

Where:
    pos     = position of token in sequence (0, 1, 2, ...)
    i       = dimension index (0, 1, 2, ..., d_model/2)
    d_model = embedding dimension (e.g. 512)
```

**Why sine and cosine?**

```
For small i (low dimensions):
    10000^(2i/512) is small → high frequency oscillation
    → captures fine-grained position differences

For large i (high dimensions):
    10000^(2i/512) is large → low frequency oscillation
    → captures coarse position differences

This gives UNIQUE encoding for every position
AND generalizes to sequences longer than seen in training
```

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denom = 10000 ** (2*i / d_model)
            PE[pos, i]   = math.sin(pos / denom)   # even dims
            PE[pos, i+1] = math.cos(pos / denom)   # odd dims

    return PE   # shape: (seq_len, d_model)

# Example
pe = positional_encoding(seq_len=10, d_model=512)
print(pe.shape)   # torch.Size([10, 512])
```

---

### Method 2 — Learned Positional Encoding (GPT-style):

Instead of a fixed formula, treat positions like words — give each position a **learnable vector** updated during training:

```python
# Learned positional encoding
max_seq_len = 1024
d_model = 512

pos_embedding = nn.Embedding(max_seq_len, d_model)

positions = torch.arange(seq_len).unsqueeze(0)  # [0, 1, 2, ..., seq_len-1]
pos_encoded = pos_embedding(positions)           # shape: (1, seq_len, 512)
```

---

### Adding to Word Embeddings:

```python
# Complete input preparation
token_ids = torch.tensor([[464, 3797, 3332]])   # (1, 3)

word_embeds = embedding_layer(token_ids)         # (1, 3, 512)
pos_encodes = positional_encoding(3, 512)        # (3, 512)

x = word_embeds + pos_encodes   # (1, 3, 512)
# NOW each vector carries:
# ✅ What the word means
# ✅ Where it appears in the sentence
```

> 💡 **Insight:** Positional encoding is like writing seat numbers on everyone's ticket before they enter a concert. Inside (the Transformer), everyone looks the same without their ticket — the seat number tells you where they belong. Without it, the model would see a crowd with no arrangement.

---

# 12. 🧠 ATTENTION MECHANISM (Timestamp: 18:46)

### From the image you shared (Embedding → Attention → Contextual Embedding):

```
Embedding (n-dim)                   Contextual Embedding (n-dim)
(no_of_tokens × embedding_dim)      (no_of_tokens × embedding_dim)

[0.012  0.819  1.190  1.129 ...]         [0.92  0.819  0.109  1.129 ...]
[0.492  0.829  0.123  0.886 ...]  →ATTN→ [0.12  0.229  0.230  0.886 ...]
[0.220  0.912  1.869  1.986 ...]         [0.220 0.912  0.169  1.986 ...]
[0.812  0.198  0.310  0.889 ...]         [0.812 0.198  0.120  0.890 ...]
...                                       ...

SAME SHAPE. DIFFERENT VALUES.
```

**Same shape in → Same shape out. But every vector is now context-aware.**

---

## 12A. THE QUERY, KEY, VALUE FRAMEWORK

Every token generates **three vectors** from its embedding:

```
Query (Q)  →  "What information am I looking for?"
Key   (K)  →  "What information do I contain?"
Value (V)  →  "What information do I actually share?"
```

**Analogy — YouTube Search:**

```
You type a search query     → Q (what you want)
Each video has tags/title   → K (what they advertise)
Each video has actual content → V (what they give you)

Search = match your Q against all K's
        → retrieve V's of best matches
        → weighted by match quality
```

---

## 12B. THE COMPLETE ATTENTION FORMULA:

```
                  QKᵀ
Attention(Q,K,V) = softmax( ─── ) · V
                             √dk
```

**Breaking it apart:**

```
Step 1:  Q = X · Wq     shape: (seq_len × dk)
         K = X · Wk     shape: (seq_len × dk)
         V = X · Wv     shape: (seq_len × dk)

Step 2:  QKᵀ            shape: (seq_len × seq_len)
         ← attention score between every pair of tokens

Step 3:  QKᵀ / √dk      ← scale to prevent vanishing gradients

Step 4:  softmax(...)   shape: (seq_len × seq_len)
         ← each row sums to 1.0 (probability distribution)

Step 5:  × V            shape: (seq_len × dk)
         ← weighted sum of value vectors
```

---

## 12C. STEP BY STEP WITH NUMBERS:

**Example sentence: "The cat sat"** (3 tokens, dk=4 for simplicity)

```
Step 1: Create Q, K, V

Q = [[0.2, 0.8, 0.1, 0.4],   ← "The" query
     [0.9, 0.1, 0.6, 0.3],   ← "cat" query
     [0.4, 0.5, 0.2, 0.7]]   ← "sat" query

K = [[0.3, 0.7, 0.2, 0.5],   ← "The" key
     [0.8, 0.2, 0.5, 0.4],   ← "cat" key
     [0.3, 0.6, 0.4, 0.8]]   ← "sat" key

Step 2: QKᵀ (raw attention scores)
         "The"  "cat"  "sat"
"The"  [  0.72   0.72   0.92 ]
"cat"  [  0.72   1.10   0.94 ]
"sat"  [  0.89   0.94   1.19 ]

Step 3: Divide by √dk = √4 = 2
         "The"  "cat"  "sat"
"The"  [  0.36   0.36   0.46 ]
"cat"  [  0.36   0.55   0.47 ]
"sat"  [  0.44   0.47   0.59 ]

Step 4: Softmax (row by row)
         "The"  "cat"  "sat"
"The"  [  0.31   0.31   0.38 ]  ← sums to 1.0
"cat"  [  0.29   0.37   0.34 ]  ← sums to 1.0
"sat"  [  0.30   0.32   0.38 ]  ← sums to 1.0

Step 5: Multiply by V
output = attention_weights × V   → shape: (3, 4)
Each token's output = weighted mix of ALL tokens' values
```

---

## 12D. COMPLETE SELF-ATTENTION IN PYTORCH:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model=512, dk=64):
        super().__init__()
        self.dk = dk
        self.Wq = nn.Linear(d_model, dk, bias=False)
        self.Wk = nn.Linear(d_model, dk, bias=False)
        self.Wv = nn.Linear(d_model, dk, bias=False)

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, d_model)
        Q = self.Wq(x)   # (batch, seq_len, dk)
        K = self.Wk(x)   # (batch, seq_len, dk)
        V = self.Wv(x)   # (batch, seq_len, dk)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.dk)
        # shape: (batch, seq_len, seq_len)

        # Apply mask (for decoder: prevent attending to future)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        # shape: (batch, seq_len, seq_len)

        output = torch.matmul(weights, V)
        # shape: (batch, seq_len, dk)

        return output, weights
```

---

## 12E. MULTI-HEAD ATTENTION:

Run attention **h times in parallel** with different weight matrices:

```
Head 1 (Wq1, Wk1, Wv1) → might focus on subject-verb relationships
Head 2 (Wq2, Wk2, Wv2) → might focus on pronoun references
Head 3 (Wq3, Wk3, Wv3) → might focus on positional proximity
...
Head 8 (Wq8, Wk8, Wv8) → might focus on semantic topics

Each head: dk = d_model / h = 512 / 8 = 64
All heads concatenated: 8 × 64 = 512 ← same as input
Then projected by Wo: (512 × 512)
```

```
MultiHead(Q,K,V) = Concat(head1,...,headh) · Wo

headi = Attention(X·Wqi, X·Wki, X·Wvi)
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, D = x.shape

        # Project and reshape into heads
        Q = self.Wq(x).view(B, S, self.num_heads, self.dk).transpose(1,2)
        K = self.Wk(x).view(B, S, self.num_heads, self.dk).transpose(1,2)
        V = self.Wv(x).view(B, S, self.num_heads, self.dk).transpose(1,2)
        # shape: (B, num_heads, S, dk)

        # Scaled dot-product per head
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        # shape: (B, num_heads, S, dk)

        # Concatenate heads
        attended = attended.transpose(1,2).contiguous().view(B, S, D)
        # shape: (B, S, D=512)

        return self.Wo(attended)
```

---

## 12F. MASKED ATTENTION (Decoder):

GPT generates text **one token at a time.** When predicting token at position 5, it must not see tokens 6, 7, 8...

```
Mask (upper triangle = -inf → becomes 0 after softmax):

         pos0   pos1   pos2   pos3
pos0  [  ✅     -inf   -inf   -inf  ]  ← can only see itself
pos1  [  ✅     ✅     -inf   -inf  ]  ← sees positions 0,1
pos2  [  ✅     ✅     ✅     -inf  ]  ← sees positions 0,1,2
pos3  [  ✅     ✅     ✅     ✅    ]  ← sees all previous
```

```python
seq_len = 4
mask = torch.tril(torch.ones(seq_len, seq_len))
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

scores = scores.masked_fill(mask == 0, float('-inf'))
```

> 💡 **Insight:** Masked attention is the Transformer's way of respecting causality — cause must come before effect, past must come before future. Without the mask, the model would "cheat" by looking at future tokens during training, and then have nothing to look at during generation. The mask enforces the fundamental law that you can only use what you've already seen.

---

# 13. 🏗️ FULL GPT ARCHITECTURE — Complete Pipeline (Timestamp: 14:46 + 18:46)

### From the architecture diagram you shared:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│     softmax                                     │
│        ↑                                        │
│     Linear  (Unembedding)                       │
│        ↑                                        │
│  ┌─────────────────────┐                        │
│  │   Add and Norm      │ ← 2nd residual+norm    │
│  │        ↑    ←───────┤                        │
│  │   Feed Forward      │                        │
│  │        ↑            │                        │
│  │   Add and Norm      │ ← 1st residual+norm    │
│  │        ↑    ←───────┤                        │
│  │ Multi Head Attention│ ←── Attention Mechanism│
│  └─────────────────────┘                        │
│      Decoder Block × N                          │
│        ↑                                        │
│  Positional Encoding (+)                        │
│        ↑                                        │
│     Embedding                                   │
│        ↑                                        │
│   Tokenized Text                                │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 13A. EACH COMPONENT EXPLAINED:

### FEED FORWARD NETWORK (inside Decoder Block):

```
Input:   (seq_len × 512)
         ↓
Linear:  W1 (512 → 2048)  ← expand 4×
         ↓
ReLU:    max(0, x)        ← non-linearity
         ↓
Linear:  W2 (2048 → 512)  ← compress back
         ↓
Output:  (seq_len × 512)
```

```
FFN(x) = ReLU(x · W1 + b1) · W2 + b2
```

```python
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))
```

> 💡 **Insight:** Attention = communication between tokens. Feed Forward = independent computation per token. After every token has gathered information from its neighbours via attention, the feed forward network processes each token individually to extract deeper patterns from that gathered information.

---

### ADD & NORM (Residual Connection + Layer Normalization):

**Residual Connection (Add):**

```
output = sublayer(x) + x
                       ↑
                 original input
                 passed through directly
```

Why? Prevents **vanishing gradients** in deep networks. With 96 layers (GPT-3), gradients would shrink to zero without shortcuts.

**Layer Normalization (Norm):**

```
For each token vector (512 numbers):
    μ = mean of all 512 values
    σ = std of all 512 values
    x_norm = (x - μ) / (σ + ε)
    output = γ · x_norm + β

Where γ, β are learned scale and shift parameters
```

```python
class AddAndNorm(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)  # Add then Norm
```

---

### COMPLETE DECODER BLOCK:

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.attention    = MultiHeadAttention(d_model, num_heads)
        self.addnorm1     = AddAndNorm(d_model)
        self.feedforward  = FeedForward(d_model, d_ff)
        self.addnorm2     = AddAndNorm(d_model)

    def forward(self, x, mask=None):
        # 1. Multi-Head Attention + residual + norm
        attn_out = self.attention(x, mask)
        x = self.addnorm1(x, attn_out)

        # 2. Feed Forward + residual + norm
        ff_out = self.feedforward(x)
        x = self.addnorm2(x, ff_out)

        return x    # shape unchanged: (B, S, d_model)
```

---

### COMPLETE GPT MODEL:

```python
class GPT(nn.Module):
    def __init__(self,
                 vocab_size=50000,
                 d_model=512,
                 num_heads=8,
                 num_layers=12,
                 d_ff=2048,
                 max_seq_len=1024):
        super().__init__()

        # Input
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding   = nn.Embedding(max_seq_len, d_model)

        # Decoder blocks × N
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Output
        self.norm      = nn.LayerNorm(d_model)
        self.unembedding = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.unembedding.weight = self.token_embedding.weight

    def forward(self, token_ids):
        B, S = token_ids.shape

        # Causal mask
        mask = torch.tril(torch.ones(S, S)).unsqueeze(0).unsqueeze(0)

        # Embedding + Positional Encoding
        positions = torch.arange(S, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.pos_embedding(positions)
        # shape: (B, S, d_model)

        # Pass through all decoder blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm
        x = self.norm(x)

        # Unembedding → logits
        logits = self.unembedding(x)        # (B, S, vocab_size)
        return logits

    def generate(self, token_ids, max_new_tokens=50):
        for _ in range(max_new_tokens):
            logits = self.forward(token_ids)
            # Take last token's prediction
            next_probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.argmax(next_probs, dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids
```

---

## 13B. COMPLETE SHAPE FLOW TABLE:

```
Stage                      Operation           Shape
────────────────────────────────────────────────────────────────────
Raw text                                       "The cat sat"
Token IDs                  Tokenizer           (B, S)
                                               e.g. (1, 3)

Token Embedding            nn.Embedding        (B, S, d_model)
                                               (1, 3, 512)

+ Positional Encoding      element-wise add    (B, S, d_model)
                                               (1, 3, 512)

─── Decoder Block ─────────────────────────────────────────────────

Multi-Head Attention       Q,K,V + softmax     (B, S, d_model)
                                               (1, 3, 512)
Add & Norm                 residual + LN       (B, S, d_model)
                                               (1, 3, 512)
Feed Forward               2 linear + ReLU     (B, S, d_model)
                                               (1, 3, 512)
Add & Norm                 residual + LN       (B, S, d_model)
                                               (1, 3, 512)

─── × N times ─────────────────────────────────────────────────────

Final Layer Norm           LN                  (B, S, d_model)
Linear (Unembedding)       matrix multiply     (B, S, vocab_size)
                                               (1, 3, 50000)
Softmax                    probabilities       (B, S, vocab_size)
                                               (1, 3, 50000)
Next Token                 argmax last pos     (B, 1)
                                               (1, 1)
```

---

## 13C. MODEL SIZES — GPT FAMILY:

```
Model       Layers(N)   d_model   Heads   Params
─────────────────────────────────────────────────
GPT-2 Small    12         768       12    117M
GPT-2 Medium   24        1024       16    345M
GPT-2 Large    36        1280       20    774M
GPT-2 XL       48        1600       25    1.5B
GPT-3          96       12288       96    175B
GPT-4         ???         ???      ???    ~1T (estimated)
```

The architecture is **identical** across all sizes. Only the numbers change.

---

# 14. 📋 MASTER SUMMARY SHEET

---

## Methods Comparison:

```
Method          Vector Type   Dim        Semantic   Context    Task
──────────────────────────────────────────────────────────────────
One-Hot         Sparse        V (50K)    ❌ No       ❌ No      Baseline
Bag of Words    Sparse        V (50K)    ❌ No       ❌ No      Doc class
TF-IDF          Sparse        V (50K)    ⚠️ Weak    ❌ No      Search
Word2Vec        Dense         300        ✅ Yes      ❌ No      NLP tasks
GloVe           Dense         300        ✅ Yes      ❌ No      NLP tasks
Transformer     Dense         512-12288  ✅ Yes      ✅ Yes     All tasks
```

---

## Key Formulas:

```
Cosine Similarity:
    cos(A,B) = A·B / (|A| × |B|)

Attention:
    Attention(Q,K,V) = softmax(QKᵀ / √dk) · V

Positional Encoding:
    PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))

Layer Normalization:
    LN(x) = γ · (x - μ) / (σ + ε) + β

Feed Forward:
    FFN(x) = ReLU(x·W1 + b1)·W2 + b2

Multi-Head:
    MH(Q,K,V) = Concat(head1,...,headh) · Wo
```

---

## Key Libraries:

```python
# Word Embeddings
from gensim.models import Word2Vec        # train Word2Vec
from sklearn.decomposition import PCA     # visualize embeddings

# Transformers from scratch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pre-trained Transformers
from transformers import GPT2Tokenizer, GPT2Model   # HuggingFace
```

---

## The 10 Things To Remember Forever:

```
1.  One-Hot → sparse, no meaning, fails at scale
2.  BoW → loses word order completely
3.  Embeddings → dense vectors, meaning = distance
4.  Word2Vec → learns embeddings by predicting context words
5.  Sliding Window → generates (target, context) training pairs
6.  CBOW → context predicts center (faster, frequent words)
7.  Skip-Gram → center predicts context (slower, better quality)
8.  Transformer Embedding → trainable, task-specific, same as row lookup
9.  Positional Encoding → sin/cos added to embeddings, gives order
10. Attention → every token looks at every other token
              → shape in = shape out
              → Embedding becomes Contextual Embedding
              → formula: softmax(QKᵀ/√dk)·V
```

---

## The Golden Thread — How It All Connects:

```
Raw Text
    ↓  tokenize
Integer Token IDs
    ↓  embedding lookup
Isolated Word Vectors          (each word knows only itself)
    ↓  + positional encoding
Position-Aware Vectors         (each word knows its location)
    ↓  multi-head attention
Contextual Vectors             (each word knows the whole sentence)
    ↓  feed forward × N layers
Deep Contextual Representations (model understands grammar+meaning+context)
    ↓  unembedding + softmax
Probability over Vocabulary    (model predicts next most likely word)
    ↓
Generated Text
```

> 💡 **Final Master Insight:** The entire journey from One-Hot Encoding to GPT is really one long answer to a single question — _"how do we give machines genuine understanding of language?"_ One-Hot said words are just labels. BoW said words are just counts. Word2Vec said words are points in space. Transformers said words are not fixed points at all — they are dynamic, context-shifting entities that mean different things in different company. That is exactly how human language works. And that is why Transformers changed everything.

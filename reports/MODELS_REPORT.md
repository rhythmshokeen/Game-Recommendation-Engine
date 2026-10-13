# GameRec: Models Report — Popularity & Content-Based Filtering (Phase 3)

**Date**: 2026-10-13  
**Project**: GameRec — End-to-End Steam Game Recommendation Engine  
**Pipeline Phase**: Phase 3 (Model A: Popularity Baseline & Model B: Content-Based Filtering)  
**Artifacts**: `models/popularity_model.joblib` (33 MB), `models/content_model.joblib` (33 MB)  

---

## 1. Model A: Popularity Baseline (Confidence-Adjusted Bayesian Shrinkage)

### Mathematical Formulation
Raw positive review ratios ($\frac{\text{positive}}{\text{total}}$) produce severe distortion for low-volume games (e.g., $1/1 = 100\%$). To correct for sample-size variance without leaking evaluation data, GameRec implements an empirical Bayesian shrinkage estimator fitted strictly on catalog training parameters:

$$W = \frac{v}{v + m} R + \frac{m}{v + m} C$$

Where:
* $R = \frac{\text{positive\_ratio}}{100}$ (Observed sample mean positive ratio).
* $v = \text{user\_reviews}$ (Total review volume recorded on Steam).
* $C = \bar{R} = 0.7705$ (Prior mean positive ratio across reviewed catalog games).
* $m = \text{Quantile}_{0.75}(v) = 206$ (Regularization support threshold based on the 75th percentile of catalog review volumes).

To balance qualitative community consensus with overall reach, a **Weighted Popularity Score** is also provided:
$$\text{PopScore} = W \cdot \log_{10}(v + 1)$$

### Features & Capabilities
1. **Overall Ranking**: Global top games ordered by `weighted_popularity` or `bayesian_rating`.
2. **Genre / Tag-Specific Slicing**: Instantaneous $O(1)$ indexed retrieval for any of the 441 community tags (e.g., *Roguelike*, *RPG*, *Sci-fi*, *Strategy*).
3. **Platform Filtering**: Restrict candidates by verified OS support (`win`, `mac`, `linux`, `steam_deck`).
4. **Exclusions**: Automatic filtering of already-seen or selected game IDs.
5. **Metadata-Backed Explanations**: Factual feedback breakdown explaining rank, positive ratio, review count, and Bayesian quality index.

### Top 5 Overall Games (Weighted Popularity Index)
1. **Counter-Strike: Global Offensive** (`app_id`: 730) — Score: $6.05$ ($88\%$ positive on $7,494,460$ reviews)
2. **Terraria** (`app_id`: 105600) — Score: $5.80$ ($97\%$ positive on $943,413$ reviews)
3. **Garry's Mod** (`app_id`: 4000) — Score: $5.69$ ($96\%$ positive on $853,733$ reviews)
4. **Wallpaper Engine** (`app_id`: 431960) — Score: $5.69$ ($98\%$ positive on $637,341$ reviews)
5. **The Witcher® 3: Wild Hunt** (`app_id`: 292030) — Score: $5.59$ ($96\%$ positive on $668,455$ reviews)

### Limitations of the Popularity Baseline
* **Popularity Bias / Matthew Effect**: Disproportionately concentrates recommendations on dominant blockbusters, starving long-tail titles.
* **Zero Personalization**: Delivers identical recommendations regardless of individual user tastes or interaction history.
* **Cold-Start Failure**: Completely excludes newly released games with low review counts, even if of exceptional quality.

---

## 2. Model B: Content-Based Filtering (Multi-Modal Sparse TF-IDF)

### Representation Architecture
Content-Based Filtering represents each game in a compact, unified multi-modal feature space combining community tags, store descriptions, and game titles without constructing an uncompressed $50,872 \times 50,872$ dense matrix ($10.35\text{ GB}$).

| Modality | Extraction Method | Weight ($w_i$) | Rationale |
| :--- | :--- | :--- | :--- |
| **Community Tags** | Unigram TF-IDF (441-tag vocabulary, atomic underscores) | **$0.60$** | Highest semantic precision for gameplay mechanics, themes, and sub-genres. |
| **Store Descriptions** | Sublinear TF-IDF (unigrams + bigrams, max 25,000 features, English stopwords) | **$0.25$** | Rich narrative and thematic context from store synopses. |
| **Game Titles** | Word n-gram TF-IDF (unigrams + bigrams) | **$0.15$** | Crucial linkage for game franchises, sequels, and DLCs; serves as robust fallback for mature games with age-gated empty tags. |

### Mathematical Alignment & Sparse Dot Product
Each modality submatrix is L2-normalized:
$$\bar{X}_{\text{tags}} = \text{L2Norm}(X_{\text{tags}}), \quad \bar{X}_{\text{desc}} = \text{L2Norm}(X_{\text{desc}}), \quad \bar{X}_{\text{title}} = \text{L2Norm}(X_{\text{title}})$$

The composite matrix is formed via weighted horizontal stacking:
$$X_{\text{combined}} = \left[ \sqrt{w_{\text{tags}}} \bar{X}_{\text{tags}} \,\|\, \sqrt{w_{\text{desc}}} \bar{X}_{\text{desc}} \,\|\, \sqrt{w_{\text{title}}} \bar{X}_{\text{title}} \right]$$

Because $\sum w_i = 1$ and each block is unit-normalized, the inner product between game $i$ and game $j$ strictly satisfies:
$$\text{CosineSim}(i, j) = X_{\text{combined}}[i] \cdot X_{\text{combined}}[j]^T = w_{\text{tags}} \cos(\text{tags}) + w_{\text{desc}} \cos(\text{desc}) + w_{\text{title}} \cos(\text{title})$$

* **In-Memory Size**: Only **$12.75\text{ MB}$** in Scipy CSR sparse matrix format.
* **Query Latency**: Top-K retrieval across all $50,872$ items executes in **$< 2\text{ ms}$** via sparse matrix-vector dot product.

### Multi-Item User Preference Profile
Given a set of user-liked games $\{g_1, \dots, g_m\}$ with optional engagement weights $\{w_1, \dots, w_m\}$ (e.g. from playtime $\ln(1 + \text{hours})$):
$$\mathbf{u} = \sum_{k=1}^m w_k X_{\text{combined}}[g_k], \quad \hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}$$

Catalog ranking is computed as $\mathbf{s} = X_{\text{combined}} \cdot \hat{\mathbf{u}}^T$, masking all previously consumed games.

---

## 3. Qualitative Validation on Real Steam Games

The system was evaluated against actual games from the dataset to inspect semantic coherence, shared metadata, and qualitative relevance:

### Test Case 1: *Tomb Raider* (`app_id`: 203160)
* **Query Themes**: Action, Adventure, Female Protagonist, Third Person, Exploration.
* **Top Recommendations**:
  1. *Rise of the Tomb Raider™* (Score: $0.570$) — Shared tags: [Action, Adventure, Female Protagonist, Singleplayer, Exploration]
  2. *Tomb Raider: Anniversary* (Score: $0.519$) — Shared tags: [Action, Adventure, Female Protagonist, Puzzle, Third Person]
  3. *Tomb Raider: Underworld* (Score: $0.493$) — Shared tags: [Action, Adventure, Female Protagonist, Third Person, Platformer]
  4. *Tomb Raider V: Chronicles* (Score: $0.473$) — Shared tags: [Action, Adventure, Female Protagonist, Platformer]
  5. *Shadow of the Tomb Raider: Definitive Edition* (Score: $0.466$) — Shared tags: [Action, Adventure, Atmospheric, Exploration, Female Protagonist]
* **Validation Assessment**: 100% franchise and thematic alignment with exact female-protagonist action-adventure continuity.

### Test Case 2: *Mount & Blade: Warband* (`app_id`: 48700)
* **Query Themes**: Medieval, Sandbox, RPG, Strategy, Open World, Historical.
* **Top Recommendations**:
  1. *Mount & Blade* (Score: $0.514$) — Shared tags: [Action, Adventure, Fantasy, Historical, Indie, Medieval, RPG]
  2. *Mount & Blade: With Fire & Sword* (Score: $0.504$) — Shared tags: [Action, Adventure, Historical, Horses, Indie, Medieval, RPG]
  3. *Mount & Blade: Warband - Napoleonic Wars* (Score: $0.502$) — Shared tags: [Action, Historical, Military, Multiplayer, War]
  4. *Mount & Blade: Warband - Viking Conquest* (Score: $0.420$) — Shared tags: [Action, Historical, Medieval, RPG, Strategy]
  5. *Kingdom Come: Deliverance – Band of Bastards* (Score: $0.377$) — Shared tags: [Action, Adventure, Historical, Horses, Medieval, RPG]
* **Validation Assessment**: Accurately surfaces related franchise entries and independently captures non-franchise thematic peers (*Kingdom Come: Deliverance*).

### Test Case 3: Multi-Game User Profile (*Mount & Blade: Warband* + *Tomb Raider*)
* **Input Profile**: Warband (Weight 2.0) + Tomb Raider (Weight 1.0)
* **Top Recommendations**:
  1. *Mount & Blade* (Profile Sim: $0.490$)
  2. *Mount & Blade: Napoleonic Wars* (Profile Sim: $0.475$)
  3. *Mount & Blade: With Fire & Sword* (Profile Sim: $0.465$)
  4. *Mount & Blade: Viking Conquest* (Profile Sim: $0.401$)
  5. *Astride* (Profile Sim: $0.354$) — Shared tags: [Adventure, Horses, Indie, Multiplayer, Open World, RPG]
* **Validation Assessment**: Blends the weighted medieval/horse themes with exploration and adventure; surfaces equine RPG *Astride* outside the immediate franchise.

---

## 4. Automated Test Suite Summary

Executed via `pytest` (`tests/test_models.py` and `tests/test_data.py`):
* **Total Tests**: **17 passed in 1.50s**.
* **Key Test Coverage**:
  - `test_popularity_recommender_fit`: Parameter calculation and shrinkage scoring.
  - `test_popularity_recommender_overall`: Metric sorting and rank order.
  - `test_popularity_recommender_tag_filter`: Genre slicing and tag index filtering.
  - `test_popularity_recommender_exclusion`: Exclusion of consumed items.
  - `test_content_recommender_fit`: Sparse matrix dimension and L2 normalization verification.
  - `test_content_recommender_similarity`: Cosine similarity ranking and shared tag extraction.
  - `test_content_recommender_user_profile`: Weighted multi-item profile construction.
  - `test_content_recommender_invalid_id`: Exception handling on unknown game IDs (`ValueError`).
  - `test_content_recommender_empty_metadata`: Non-crashing fallback on games with empty descriptions and tags.
  - `test_content_recommender_no_duplicate_results`: Guarantee of unique recommendation IDs.
  - `test_model_serialization`: Joblib save and reload fidelity for both models.

---

## 5. Model C & D: Collaborative Filtering Models (Phase 4)

### Signal Audit & Interaction Semantics
A rigorous inspection of `interactions_benchmark.parquet` (2,041,850 rows across 99,997 active users and 8,599 games) confirmed the following signals:
* **`is_recommended`** (`Boolean`): Binary recommendation verdict ($83.96\%$ True, $16.04\%$ False).
* **`hours`** (`Float32`): Playtime hours ($0.0$ to $1000.0$, median: $17.2$, mean: $66.2$).
* **`implicit_confidence`** (`Float32`): Log-transformed engagement weight $\ln(1 + \text{hours})$.
* **`date`** (`Date`): Interaction submission timestamp ($2010-10-15$ to $2022-12-31$).
* **Critical Distinction**: The Steam dataset contains **no 1-5 star numerical ratings**. Fabricating explicit ratings would violate methodology rules. Therefore, GameRec uses an **implicit feedback framework** (Hu, Koren, Volinsky):
  - Binary preference: $p_{ui} = 1$ when recommended.
  - Confidence weight: $c_{ui} = 1.0 + \alpha \ln(1 + \text{hours})$ (with $\alpha = 0.5$).
  - Negative reviews (`is_recommended = False`) are excluded from positive candidate generation.
  - Missing entries remain neutral unobserved candidates (never imputed as negative).

### Data Splitting: Zero-Leakage Temporal Holdout
To prevent temporal data leakage (predicting past actions with future interactions):
* User interactions were ordered chronologically by `(user_id, date, review_id)`.
* For all 99,928 users with $\ge 5$ interactions, the **latest review was held out into `test_interactions.parquet`** ($99,928$ interactions, $1.46\text{ MB}$).
* The prior interactions formed **`train_interactions.parquet`** ($1,941,707$ interactions, $23.11\text{ MB}$).
* All models and latent factors were fitted strictly on `train_interactions.parquet`.

### Model C: Item-Item Collaborative Filtering (`ItemItemCollaborativeRecommender`)
* **Algorithm**: Cosine similarity across normalized item vectors with **Bayesian Shrinkage Regularization** ($\lambda = 15.0$) and top-100 neighbor pruning:
  $$S_{ij} = \frac{\mathbf{r}_{*i} \cdot \mathbf{r}_{*j}}{\|\mathbf{r}_{*i}\|_2 \|\mathbf{r}_{*j}\|_2 + \lambda}$$
* **Prediction**: $\hat{\mathbf{y}}_u = \mathbf{r}_u S$.
* **In-Memory Size**: Only **$7.5\text{ MB}$** in sparse CSR format.
* **Explanation**: Transparently identifies the top contributing game in user history:
  $$\text{Explanation Item} = \arg\max_{i \in \mathcal{H}_u} \left( r_{ui} \cdot S_{ij} \right)$$
* **Training Time**: **$3.29\text{ seconds}$** on 1.94M training interactions.

### Model D: Matrix Factorization (`MatrixFactorizationRecommender`)
* **Algorithm**: Truncated Singular Value Decomposition (TruncatedSVD) with $d = 64$ latent components (`random_state=42`) on the L2-normalized user interaction matrix:
  $$R_{\text{norm}} \approx P Q^T$$
  where $Q \in \mathbb{R}^{8,599 \times 64}$ are the normalized item latent factors.
* **Prediction**: For any user $u$ with positive interaction vector $\mathbf{r}_u$:
  $$\mathbf{p}_u = \text{L2Norm}(\mathbf{r}_u Q), \quad \hat{\mathbf{y}}_u = Q \mathbf{p}_u^T$$
* **In-Memory Size**: Item factors $8,599 \times 64 \times 4\text{ bytes} \approx \mathbf{2.2\text{ MB}}$.
* **Training Time**: **$2.52\text{ seconds}$**.

### Comparative Analysis: Neighborhood vs. Matrix Factorization

| Dimension | Item-Item Collaborative Filtering | Matrix Factorization (SVD) |
| :--- | :--- | :--- |
| **Recommendation Nature** | Anchor-driven, high-precision local neighborhood | Serendipitous, global latent taste clusters |
| **Explanation Capability** | **High** ("Recommended because you played X (co-occurrence: 0.30)") | **Moderate** (Latent factor projection affinity) |
| **Transitive Relations** | Limited to direct co-occurrence paths | **Strong** (Uncovers shared latent dimensions across disjoint games) |
| **Popularity Bias** | Moderately influenced by popular item co-occurrence | Lower; surfaces relevant mid-tail and niche games |
| **Inference Latency** | $< 1\text{ ms}$ (sparse vector dot product) | $< 1\text{ ms}$ (dense projection dot product) |

#### What Worked:
1. **Item-Item CF Precision**: For User `550` (who played *Rise of the Tomb Raider*, *The Witcher 3*, and *Gunfire Reborn*), Item-Item CF correctly identified *Tomb Raider*, *Shadow of the Tomb Raider*, *Cyberpunk 2077*, and *Hades* with explicit attribution to specific played games.
2. **SVD Serendipity & Latent Clusters**: For User `4002` (who played *Stick Fight*, *Rust*, and *Bloons TD*), SVD uncovered the latent cluster of chaotic indie multiplayer/party games (*Crab Game*, *Krunker*, *Shotgun Farmers*, *Paunch*), capturing nuanced user intent beyond direct co-occurrence.
3. **Efficiency**: Both models train in under 4 seconds and occupy $<10\text{ MB}$ of RAM, completely eliminating out-of-memory risks.

#### What Failed & Necessary Adjustments:
1. **Catalog Cold-Start**: Neither collaborative filtering model can recommend games outside the benchmark training matrix (i.e. games with $< 20$ reviews or unreviewed catalog titles).
   - *Fix*: The upcoming Hybrid Model will seamlessly route cold-start games through Content-Based and Popularity filtering.
2. **Dominant Head Items in Neighborhood CF**: Without shrinkage regularization ($\lambda = 15.0$), ubiquitous games like *Wallpaper Engine* and *Counter-Strike* appeared in almost every neighborhood list. Shrinkage successfully suppresses spurious co-occurrence from low-support pairs while preserving genuine high-affinity connections.

---

## 6. Model E: Hybrid Recommendation Engine (Phase 5)

### Architecture & Mathematical Formulation
The Hybrid Recommendation Engine integrates Content-Based Filtering, Collaborative Filtering (Item-Item and SVD), and Bayesian Popularity Baselines into a unified, calibrated scoring framework:

$$\text{Score}_{\text{hybrid}}(i) = \alpha \cdot \tilde{S}_{\text{content}}(i) + (1 - \alpha) \cdot \tilde{S}_{\text{collab}}(i)$$

Where:
* $\tilde{S}_{\text{content}}(i) \in [0, 1]$: Min-max scaled cosine similarity over multi-modal TF-IDF features (community tags, store description, title tokens).
* $\tilde{S}_{\text{collab}}(i) \in [0, 1]$: Min-max scaled collaborative affinity from Item-Item co-occurrence shrinkage or Latent Factor SVD projection.
* $\alpha \in [0.0, 1.0]$: Configurable weight controlling the content-collaborative trade-off:
  - $\alpha = 1.0$: Pure Content-Based Filtering (independent of collaborative history).
  - $\alpha = 0.0$: Pure Collaborative Filtering (confined to community co-play patterns).
  - $0.0 < \alpha < 1.0$: Weighted Hybrid Blend (Default: $\alpha = 0.50$).

### Explicit Cold-Start Routing Rules
1. **Completely New User ($0$ interactions)**:
   - Routed to the **Bayesian Popularity Baseline** ($W$). Recommends globally validated high-quality titles with sample-size regularization ($m = 206$).
2. **Limited History User ($< 3$ interactions)**:
   - Routed to **Content-Based Profile Matching** using their limited played games, with popularity fallback for ties.
3. **Warm User ($\ge 3$ interactions)**:
   - Routed to full **Calibrated Hybrid Ranking**.
4. **Cold-Start Item (Unreviewed or $< 20$ reviews)**:
   - Evaluated via Content-Based similarity (collaborative score defaults to 0). Never excluded merely due to lack of community history.

### Diversity Control & Redundancy Suppression (MMR)
Greedy top-K ranking in content systems frequently produces near-duplicate recommendations (e.g. 5 consecutive entries from the *Tomb Raider* series).
The engine integrates **Maximal Marginal Relevance (MMR)** and franchise caps:

$$i^* = \arg\max_{i \in \mathcal{C} \setminus \mathcal{S}} \left[ \lambda_{\text{div}} \cdot \text{Rel}(i) - (1 - \lambda_{\text{div}}) \max_{j \in \mathcal{S}} \text{Sim}(i, j) \right]$$

* **$\lambda_{\text{div}} = 0.70$**: Balances high relevance with novelty.
* **$\text{Max Franchises} = 2$**: Caps any single franchise prefix at 2 entries in the top-K list.
* **Demonstrated Impact**: For *Tomb Raider*, greedy ranking produced 5 *Tomb Raider* games. MMR diversity capped the series at 2 and surfaced high-relevance, diverse peer action-adventure titles (*Batman: Arkham City*, *Life is Strange*, *DmC: Devil May Cry*).

### Truthful & Calibrated Explanations
Explanations are strictly transparent and factual, reporting:
* Exact component contributions: $\text{Score}_{\text{hybrid}}$, $S_{\text{content}}$, $S_{\text{collab}}$, and active $\alpha$.
* Contributing historical titles (e.g. *"Recommended because you played 'The Witcher 3'"*).
* Shared community tags (e.g. *"[Open World, RPG, Fantasy]"*).
* Avoids fabricated "probabilities of enjoyment".



"""
GameRec: Interactive Streamlit Recommendation Engine Application.
Provides a modern, dark-themed gaming discovery interface powered by
the evaluated GameRec recommendation algorithms:
- Bayesian Popularity Baseline
- Multi-Modal Content-Based Filtering
- Item-Item Collaborative Filtering (Bayesian Shrinkage)
- Latent Factor Matrix Factorization (TruncatedSVD)
- Configurable Hybrid Recommendation Engine
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gamerec.app")

import sys

# Define Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Add src to pythonpath so gamerec is always importable regardless of environment
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GameRec | Steam Recommendation Engine",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Modern Gaming Design System (CSS)
# ---------------------------------------------------------------------------
STEAM_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e3e9f0;
}

/* Base App Dark Background */
.stApp {
    background: radial-gradient(circle at 15% 15%, #121c29 0%, #0a0e14 60%, #06090d 100%);
}

/* Card Design */
.gamerec-card {
    background: linear-gradient(145deg, rgba(27, 40, 56, 0.85) 0%, rgba(16, 25, 36, 0.85) 100%);
    border: 1px solid rgba(102, 192, 244, 0.2);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.gamerec-card:hover {
    transform: translateY(-4px);
    border-color: rgba(102, 192, 244, 0.55);
    box-shadow: 0 12px 30px rgba(0, 229, 255, 0.15);
}

/* Badge Styles */
.badge-pill {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 16px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
    letter-spacing: 0.3px;
}

.badge-tag {
    background: rgba(102, 192, 244, 0.15);
    color: #66c0f4;
    border: 1px solid rgba(102, 192, 244, 0.3);
}

.badge-positive {
    background: rgba(46, 204, 113, 0.18);
    color: #2ecc71;
    border: 1px solid rgba(46, 204, 113, 0.35);
}

.badge-score {
    background: rgba(243, 156, 18, 0.18);
    color: #f39c12;
    border: 1px solid rgba(243, 156, 18, 0.35);
}

.badge-price {
    background: rgba(155, 89, 182, 0.2);
    color: #c39bd3;
    border: 1px solid rgba(155, 89, 182, 0.35);
}

.badge-platform {
    background: rgba(255, 255, 255, 0.08);
    color: #bdc3c7;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

/* Rank Circle */
.rank-badge {
    background: linear-gradient(135deg, #1999ff 0%, #00e5ff 100%);
    color: #06090d;
    font-weight: 800;
    font-size: 0.85rem;
    padding: 3px 10px;
    border-radius: 8px;
    display: inline-block;
    margin-bottom: 6px;
}

/* Explanation Box */
.explanation-box {
    background: rgba(0, 0, 0, 0.35);
    border-left: 3px solid #66c0f4;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #c5d3e0;
    margin-top: 10px;
    line-height: 1.4;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #0c1219;
    border-right: 1px solid rgba(102, 192, 244, 0.15);
}

/* Metric Display */
div[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif;
    color: #66c0f4;
    font-weight: 700;
}
</style>
"""
st.markdown(STEAM_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Resource Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading GameRec recommendation models...")
def load_models():
    """Loads all persisted model artifacts once per application session."""
    logger.info("Initializing models from models/ directory...")
    pop_path = MODELS_DIR / "popularity_model.joblib"
    content_path = MODELS_DIR / "content_model.joblib"
    collab_path = MODELS_DIR / "item_collaborative_model.joblib"
    svd_path = MODELS_DIR / "matrix_factorization_model.joblib"
    config_path = MODELS_DIR / "hybrid_config.json"

    if not pop_path.exists() or not content_path.exists() or not collab_path.exists():
        st.error(
            "Model artifacts missing! Please ensure baseline and collaborative models "
            "have been trained and saved to the `models/` directory."
        )
        st.stop()

    pop_model = joblib.load(pop_path)
    content_model = joblib.load(content_path)
    collab_model = joblib.load(collab_path)
    svd_model = joblib.load(svd_path) if svd_path.exists() else None

    # Load Hybrid configuration
    default_alpha = 0.5
    diversity_lambda = 0.7
    max_per_franchise = 2
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                h_cfg = json.load(f)
                default_alpha = h_cfg.get("default_alpha", 0.5)
                diversity_lambda = h_cfg.get("diversity_lambda", 0.7)
                max_per_franchise = h_cfg.get("max_per_franchise", 2)
        except Exception as e:
            logger.warning(f"Could not read hybrid_config.json: {e}")

    # Import HybridRecommender dynamically
    from gamerec.hybrid_model import HybridRecommender

    hybrid_model = HybridRecommender(
        content_model=content_model,
        collaborative_model=collab_model,
        popularity_model=pop_model,
        default_alpha=default_alpha,
        diversity_lambda=diversity_lambda,
        max_per_franchise=max_per_franchise,
    )

    return {
        "popularity": pop_model,
        "content": content_model,
        "collaborative": collab_model,
        "svd": svd_model,
        "hybrid": hybrid_model,
    }


@st.cache_data(show_spinner="Loading Steam game catalog...")
def load_catalog_data() -> pl.DataFrame:
    """Loads unified games catalog metadata from parquet."""
    catalog_path = DATA_DIR / "games_catalog.parquet"
    if not catalog_path.exists():
        st.error(f"Catalog file not found at {catalog_path}.")
        st.stop()
    return pl.read_parquet(catalog_path)


@st.cache_data(show_spinner="Loading benchmark evaluation metrics...")
def load_evaluation_metrics() -> Optional[Dict[str, Any]]:
    """Loads saved offline evaluation metrics from reports directory."""
    eval_json_path = REPORTS_DIR / "evaluation_metrics.json"
    if not eval_json_path.exists():
        return None
    try:
        with open(eval_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load evaluation metrics: {e}")
        return None


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "user_library" not in st.session_state:
    # Library format: {app_id: {"title": str, "status": "liked"|"played"|"disliked", "hours": float}}
    st.session_state.user_library = {}

if "demo_profile_name" not in st.session_state:
    st.session_state.demo_profile_name = "Custom Guest Session"


def reset_library():
    st.session_state.user_library = {}
    st.session_state.demo_profile_name = "Custom Guest Session"
    st.toast("User library successfully reset to clean slate!", icon="🧹")


def add_to_library(app_id: int, title: str, status: str = "liked", hours: float = 10.0):
    st.session_state.user_library[int(app_id)] = {
        "title": title,
        "status": status,
        "hours": hours,
    }
    st.toast(f"Added '{title}' as {status}!", icon="🎮")


def remove_from_library(app_id: int):
    if int(app_id) in st.session_state.user_library:
        title = st.session_state.user_library[int(app_id)]["title"]
        del st.session_state.user_library[int(app_id)]
        st.toast(f"Removed '{title}' from library.", icon="🗑️")


# ---------------------------------------------------------------------------
# Helper Render Functions
# ---------------------------------------------------------------------------
def get_header_image_url(app_id: int) -> str:
    """Returns official Steam CDN header image URL."""
    return f"https://shared.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg"


def render_game_card(
    item: Dict[str, Any],
    rank: Optional[int] = None,
    show_actions: bool = True,
    col_width: int = 12,
):
    """Renders a modern, responsive game card with Steam aesthetics."""
    aid = int(item["app_id"])
    title = item.get("title", f"Game #{aid}")
    score = item.get("score", item.get("hybrid_score", item.get("similarity_score", None)))
    pos_ratio = item.get("positive_ratio", None)
    user_reviews = item.get("user_reviews", None)
    price = item.get("price", None)
    tags = item.get("all_tags", item.get("tags", []))
    explanation = item.get("explanation", None)

    # Format price
    if price is None or price == 0.0:
        price_str = "Free to Play"
    else:
        price_str = f"${price:.2f}"

    # Build card HTML
    rank_html = f'<div class="rank-badge">#{rank} TOP MATCH</div>' if rank else ""
    score_html = f'<span class="badge-pill badge-score">★ Score: {score:.3f}</span>' if score is not None else ""
    pos_html = f'<span class="badge-pill badge-positive">👍 {pos_ratio}% Positive</span>' if pos_ratio is not None else ""
    price_html = f'<span class="badge-pill badge-price">{price_str}</span>'
    
    # Review volume
    reviews_html = ""
    if user_reviews is not None and user_reviews > 0:
        reviews_html = f'<span class="badge-pill badge-platform">👥 {user_reviews:,} reviews</span>'

    # Tags pills (top 4)
    tags_html = ""
    if tags:
        clean_tags = tags[:4] if isinstance(tags, list) else []
        tags_html = " ".join([f'<span class="badge-pill badge-tag">{t}</span>' for t in clean_tags])

    # Explanation block
    exp_html = ""
    if explanation:
        exp_html = f'<div class="explanation-box">{explanation}</div>'

    card_html = f"""
    <div class="gamerec-card">
        {rank_html}
        <h4 style="margin: 0 0 8px 0; color: #ffffff; font-weight: 700; font-size: 1.15rem;">{title}</h4>
        <div style="margin-bottom: 8px;">
            {score_html}
            {pos_html}
            {reviews_html}
            {price_html}
        </div>
        <div style="margin-bottom: 8px;">
            {tags_html}
        </div>
        {exp_html}
    </div>
    """

    col_img, col_info = st.columns([1, 2.5])
    with col_img:
        img_url = get_header_image_url(aid)
        st.image(
            img_url,
            use_container_width=True,
            caption=f"App ID: {aid}",
        )
    with col_info:
        st.markdown(card_html, unsafe_allow_html=True)
        if show_actions:
            act_c1, act_c2, act_c3 = st.columns([1, 1, 1])
            with act_c1:
                if st.button("❤️ Like", key=f"like_{aid}_{rank}_{np.random.randint(100000)}"):
                    add_to_library(aid, title, status="liked")
                    st.rerun()
            with act_c2:
                if st.button("🎮 Played", key=f"play_{aid}_{rank}_{np.random.randint(100000)}"):
                    add_to_library(aid, title, status="played")
                    st.rerun()
            with act_c3:
                if st.button("🚫 Dismiss", key=f"dis_{aid}_{rank}_{np.random.randint(100000)}"):
                    add_to_library(aid, title, status="disliked")
                    st.rerun()


# ---------------------------------------------------------------------------
# App Main Execution & Navigation
# ---------------------------------------------------------------------------
def main():
    models = load_models()
    catalog_df = load_catalog_data()
    eval_metrics = load_evaluation_metrics()

    # Pre-compute title search dictionary
    all_titles = catalog_df["title"].to_list()
    all_app_ids = catalog_df["app_id"].to_list()
    title_to_aid = {t: int(a) for t, a in zip(all_titles, all_app_ids)}
    aid_to_title = {int(a): t for t, a in zip(all_titles, all_app_ids)}

    # Sidebar Header & Brand
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h1 style="color: #66c0f4; font-size: 1.8rem; margin: 0; font-weight: 800; letter-spacing: -0.5px;">
                🎮 GameRec
            </h1>
            <p style="color: #8f98a0; font-size: 0.85rem; margin-top: 4px;">
                Steam Recommendation Engine
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_selection = st.sidebar.radio(
        "Navigation",
        [
            "🌟 Discover & Search",
            "🎯 Similar Games",
            "👤 Personalized Library",
            "⚖️ Model Comparison",
            "📊 Offline Evaluation",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Sidebar: User Profile & Session State
    st.sidebar.subheader("👤 Demo Taste Profile")
    st.sidebar.caption(f"Active Profile: **{st.session_state.demo_profile_name}**")
    
    lib_count = len(st.session_state.user_library)
    st.sidebar.write(f"Games in Library: **{lib_count}**")

    # Quick Preset Profiles
    preset_choice = st.sidebar.selectbox(
        "Load Preset Gamer Taste:",
        [
            "-- Choose Preset --",
            "Cold Start (0 Games)",
            "RPG & Open World (Witcher, Skyrim)",
            "Competitive FPS (CS:GO, R6 Siege)",
            "Indie Roguelike (Hades, Dead Cells)",
            "Action Adventure (Tomb Raider, Arkham)",
        ],
    )

    if preset_choice == "Cold Start (0 Games)":
        if st.session_state.demo_profile_name != "Cold Start":
            reset_library()
            st.session_state.demo_profile_name = "Cold Start"
            st.rerun()

    elif preset_choice == "RPG & Open World (Witcher, Skyrim)":
        if st.session_state.demo_profile_name != "RPG & Open World":
            st.session_state.user_library = {
                292030: {"title": "The Witcher 3: Wild Hunt", "status": "liked", "hours": 95.0},
                489830: {"title": "The Elder Scrolls V: Skyrim Special Edition", "status": "liked", "hours": 120.0},
                435150: {"title": "Divinity: Original Sin 2", "status": "liked", "hours": 65.0},
                22380: {"title": "Fallout: New Vegas", "status": "played", "hours": 40.0},
            }
            st.session_state.demo_profile_name = "RPG & Open World"
            st.rerun()

    elif preset_choice == "Competitive FPS (CS:GO, R6 Siege)":
        if st.session_state.demo_profile_name != "Competitive FPS":
            st.session_state.user_library = {
                730: {"title": "Counter-Strike: Global Offensive", "status": "liked", "hours": 250.0},
                359550: {"title": "Tom Clancy's Rainbow Six Siege", "status": "liked", "hours": 110.0},
                440: {"title": "Team Fortress 2", "status": "played", "hours": 45.0},
                578080: {"title": "PUBG: BATTLEGROUNDS", "status": "liked", "hours": 80.0},
            }
            st.session_state.demo_profile_name = "Competitive FPS"
            st.rerun()

    elif preset_choice == "Indie Roguelike (Hades, Dead Cells)":
        if st.session_state.demo_profile_name != "Indie Roguelike":
            st.session_state.user_library = {
                1145360: {"title": "Hades", "status": "liked", "hours": 55.0},
                588650: {"title": "Dead Cells", "status": "liked", "hours": 40.0},
                250900: {"title": "The Binding of Isaac: Rebirth", "status": "liked", "hours": 130.0},
                367520: {"title": "Hollow Knight", "status": "liked", "hours": 35.0},
            }
            st.session_state.demo_profile_name = "Indie Roguelike"
            st.rerun()

    elif preset_choice == "Action Adventure (Tomb Raider, Arkham)":
        if st.session_state.demo_profile_name != "Action Adventure":
            st.session_state.user_library = {
                203160: {"title": "Tomb Raider", "status": "liked", "hours": 22.0},
                208650: {"title": "Batman: Arkham City - Game of the Year Edition", "status": "liked", "hours": 30.0},
                319630: {"title": "Life is Strange™", "status": "liked", "hours": 18.0},
            }
            st.session_state.demo_profile_name = "Action Adventure"
            st.rerun()

    if st.sidebar.button("🧹 Clear User Library", use_container_width=True):
        reset_library()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Catalog: **50,872** games | Memory: Compressed Sparse CSR")
    st.sidebar.caption("Evaluation: Leak-free temporal holdout")

    # =========================================================================
    # PAGE 1: DISCOVER & SEARCH
    # =========================================================================
    if nav_selection == "🌟 Discover & Search":
        st.title("🌟 Discover Steam Games")
        st.write(
            "Explore the catalog using confidence-weighted Bayesian popularity rankings, "
            "genre tags, platform compatibility, and instant title search."
        )

        disc_tab1, disc_tab2 = st.tabs(["🔥 Top Ranked Popular Games", "🔍 Search Catalog"])

        with disc_tab1:
            # Filters
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                available_tags = ["All Genres"] + models["popularity"].get_available_tags(min_games=30)
                selected_tag = st.selectbox("Genre / Community Tag:", available_tags)
            with f_col2:
                selected_platform = st.selectbox(
                    "Platform:",
                    ["Any Platform", "Windows (win)", "macOS (mac)", "Linux (linux)", "Steam Deck Verified"],
                )
            with f_col3:
                ranking_metric = st.selectbox(
                    "Ranking Method:",
                    [
                        "Weighted Popularity (Bayesian * Vol)",
                        "Bayesian Quality Rating Only",
                        "Raw Review Volume",
                    ],
                )
            with f_col4:
                top_k = st.slider("Games to display:", min_value=5, max_value=25, value=10, step=5)

            # Map inputs
            tag_filter = None if selected_tag == "All Genres" else selected_tag
            plat_filter = None
            if "win" in selected_platform:
                plat_filter = "win"
            elif "mac" in selected_platform:
                plat_filter = "mac"
            elif "linux" in selected_platform:
                plat_filter = "linux"
            elif "Deck" in selected_platform:
                plat_filter = "steam_deck"

            metric_code = "weighted_popularity"
            if "Quality" in ranking_metric:
                metric_code = "bayesian_rating"
            elif "Raw" in ranking_metric:
                metric_code = "user_reviews"

            # Seen exclusion
            excluded_seen = list(st.session_state.user_library.keys())

            with st.spinner("Retrieving popular titles..."):
                popular_recs = models["popularity"].recommend(
                    k=top_k,
                    tag=tag_filter,
                    platform=plat_filter,
                    ranking_metric=metric_code,
                    exclude_app_ids=excluded_seen,
                )

            if not popular_recs:
                st.info("No games matched the selected filter criteria. Try broadening your filters.")
            else:
                st.subheader(f"Top {len(popular_recs)} Games in {selected_tag}")
                for item in popular_recs:
                    render_game_card(item, rank=item["rank"], show_actions=True)

        with disc_tab2:
            st.subheader("Search Steam Catalog")
            search_query = st.text_input(
                "Type a game title to search:",
                placeholder="e.g. Witcher, Portal, Half-Life, Terraria...",
            )

            if search_query:
                # Polars substring search
                matches = catalog_df.filter(
                    pl.col("title").str.to_lowercase().str.contains(search_query.strip().lower())
                ).sort("user_reviews", descending=True).head(12)

                if matches.shape[0] == 0:
                    st.warning(f"No games found matching '{search_query}'.")
                else:
                    st.write(f"Found **{matches.shape[0]}** matching titles:")
                    for row in matches.to_dicts():
                        item_dict = {
                            "app_id": row["app_id"],
                            "title": row["title"],
                            "positive_ratio": row["positive_ratio"],
                            "user_reviews": row["user_reviews"],
                            "price": row["price_final"],
                            "tags": row["tags"],
                            "explanation": (
                                f"Matches search '{search_query}'. "
                                f"Catalog stats: {row['positive_ratio']}% positive on {row['user_reviews']:,} reviews."
                            ),
                        }
                        render_game_card(item_dict, show_actions=True)

    # =========================================================================
    # PAGE 2: SIMILAR GAMES
    # =========================================================================
    elif nav_selection == "🎯 Similar Games":
        st.title("🎯 Find Games Similar to a Selected Title")
        st.write(
            "Uses multi-modal TF-IDF representations (Steam tags, store descriptions, and franchise title patterns) "
            "with Maximal Marginal Relevance (MMR) diversity reranking to surface non-redundant peer recommendations."
        )

        col_sel, col_cfg = st.columns([2, 1])
        with col_sel:
            # Popular default suggestions
            default_options = [
                "The Witcher 3: Wild Hunt",
                "Tomb Raider",
                "Portal 2",
                "Terraria",
                "Hades",
                "Counter-Strike: Global Offensive",
                "Stardew Valley",
            ]
            selected_title = st.selectbox(
                "Select or search for a Seed Game:",
                options=all_titles,
                index=all_titles.index("The Witcher 3: Wild Hunt") if "The Witcher 3: Wild Hunt" in all_titles else 0,
            )
        with col_cfg:
            k_sim = st.slider("Number of recommendations:", 5, 20, 10, 5)
            use_diversity = st.checkbox("Apply MMR Diversity & Franchise Capping", value=True)

        seed_aid = title_to_aid.get(selected_title)
        if seed_aid:
            # Show Seed Game Info Card
            seed_row = catalog_df.filter(pl.col("app_id") == seed_aid).to_dicts()[0]
            st.markdown("### 📌 Selected Seed Game")
            col_s1, col_s2 = st.columns([1, 3])
            with col_s1:
                st.image(get_header_image_url(seed_aid), use_container_width=True)
            with col_s2:
                seed_price = "Free to Play" if seed_row["price_final"] == 0 else f"${seed_row['price_final']:.2f}"
                seed_tags = ", ".join(seed_row["tags"][:8]) if seed_row["tags"] else "N/A"
                st.markdown(
                    f"**{seed_row['title']}** (App ID: `{seed_aid}`)  \n"
                    f"⭐ {seed_row['positive_ratio']}% Positive ({seed_row['user_reviews']:,} reviews) | "
                    f"💲 {seed_price}  \n"
                    f"**Key Tags**: {seed_tags}"
                )

            st.markdown("---")
            st.subheader(f"Top Recommendations for Fans of '{selected_title}'")

            # Exclude seed game and user library
            exclude_ids = set(st.session_state.user_library.keys())
            exclude_ids.add(seed_aid)

            with st.spinner("Computing cosine similarity across 50k titles..."):
                raw_recs = models["content"].recommend_similar(
                    app_id=seed_aid,
                    k=k_sim * 2 if use_diversity else k_sim,
                    exclude_seen=exclude_ids,
                )

                if use_diversity:
                    from gamerec.ranking import maximal_marginal_relevance
                    similar_recs = maximal_marginal_relevance(
                        candidates=raw_recs,
                        feature_matrix=models["content"].feature_matrix,
                        app_to_idx=models["content"].app_id_to_idx,
                        diversity_lambda=0.70,
                        top_k=k_sim,
                        max_per_franchise=2,
                    )
                else:
                    similar_recs = raw_recs[:k_sim]

            if not similar_recs:
                st.info("No similar games found meeting the criteria.")
            else:
                for item in similar_recs:
                    render_game_card(item, rank=item["rank"], show_actions=True)

    # =========================================================================
    # PAGE 3: PERSONALIZED RECOMMENDATIONS
    # =========================================================================
    elif nav_selection == "👤 Personalized Library":
        st.title("👤 Personalized Taste Profile Recommendations")
        st.write(
            "Build your Steam library by adding games you love, have played, or dislike. "
            "The Hybrid Engine balances Content-Based theme matching with Collaborative co-play patterns."
        )

        # Profile Overview
        col_prof1, col_prof2 = st.columns([2, 1])
        with col_prof1:
            st.subheader("Your Library & Preferences")
            if not st.session_state.user_library:
                st.info(
                    "Your library is currently empty (Cold Start). "
                    "Pick a preset from the sidebar or search games below to add them!"
                )
            else:
                # Show active library items in pills/expander
                with st.expander(f"📚 View Active Library ({len(st.session_state.user_library)} titles)", expanded=True):
                    for aid, meta in list(st.session_state.user_library.items()):
                        l_c1, l_c2, l_c3 = st.columns([3, 1, 1])
                        with l_c1:
                            st.write(f"🎮 **{meta['title']}** (Status: `{meta['status']}`, {meta['hours']} hrs)")
                        with l_c2:
                            pass
                        with l_c3:
                            if st.button("Remove", key=f"del_{aid}"):
                                remove_from_library(aid)
                                st.rerun()

        with col_prof2:
            st.subheader("⚙️ Recommendation Controls")
            alpha_val = st.slider(
                "Hybrid Weight (α):",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help=(
                    "α = 1.0: Pure Content-Based (Thematic tags & store text).\n"
                    "α = 0.0: Pure Collaborative Filtering (Player co-occurrence).\n"
                    "α = 0.5: Calibrated Balanced Blend."
                ),
            )
            # Explain alpha state
            if alpha_val == 1.0:
                st.caption("🔵 **Mode**: Pure Content-Based Filtering")
            elif alpha_val == 0.0:
                st.caption("🟢 **Mode**: Pure Collaborative Filtering")
            else:
                st.caption(f"🟣 **Mode**: Hybrid Blend ({int(alpha_val*100)}% Content / {int((1-alpha_val)*100)}% Collab)")

            div_toggle = st.checkbox("Enable MMR Diversity Reranking", value=True)
            k_personal = st.slider("Top K Recommendations:", 5, 20, 10, 5)

        # Add Game to Library Interface
        st.markdown("---")
        st.subheader("➕ Add a Game to Your Library")
        add_c1, add_c2, add_c3, add_c4 = st.columns([3, 1, 1, 1])
        with add_c1:
            add_title = st.selectbox("Search game to add:", options=["-- Select Game --"] + all_titles)
        with add_c2:
            add_status = st.selectbox("Interaction:", ["liked", "played", "disliked"])
        with add_c3:
            add_hours = st.number_input("Hours Played:", min_value=1.0, max_value=2000.0, value=25.0, step=5.0)
        with add_c4:
            st.write("")
            st.write("")
            if st.button("Add Game", use_container_width=True):
                if add_title != "-- Select Game --" and add_title in title_to_aid:
                    add_to_library(title_to_aid[add_title], add_title, add_status, add_hours)
                    st.rerun()

        # Generate Recommendations
        st.markdown("---")
        st.subheader("🎯 Recommended For You")

        library_items = [
            {"app_id": aid, "weight": 1.0 if meta["status"] == "liked" else (0.5 if meta["status"] == "played" else -0.5)}
            for aid, meta in st.session_state.user_library.items()
            if meta["status"] in ["liked", "played"]
        ]
        excluded_ids = set(st.session_state.user_library.keys())

        if not library_items:
            # Explicit Cold-Start Routing Banner
            st.warning(
                "🚨 **Cold-Start Fallback Active**: No library history detected. "
                "Routing automatically to the Bayesian Popularity Baseline with shrinkage regularization."
            )
            recs = models["hybrid"]._handle_cold_start_user(k=k_personal, exclude_seen=excluded_ids)
        elif len(library_items) < 3:
            # Limited History Routing Banner
            st.info(
                f"ℹ️ **Limited History Routing Active ({len(library_items)} items)**: "
                f"Fewer than 3 games registered. Generating recommendations via Content-Based profile matching."
            )
            recs = models["hybrid"].recommend_interactive(
                seed_games=library_items,
                k=k_personal,
                alpha=alpha_val,
                exclude_seen=excluded_ids,
                use_diversity=div_toggle,
            )
        else:
            # Full Warm User Hybrid
            st.success(
                f"✅ **Full Hybrid Engine Active ({len(library_items)} items)**: "
                f"Blending Content Modalities and Collaborative Affinity with α={alpha_val:.2f}."
            )
            recs = models["hybrid"].recommend_interactive(
                seed_games=library_items,
                k=k_personal,
                alpha=alpha_val,
                exclude_seen=excluded_ids,
                use_diversity=div_toggle,
            )

        if recs:
            for item in recs:
                render_game_card(item, rank=item.get("rank"), show_actions=True)

    # =========================================================================
    # PAGE 4: MODEL COMPARISON
    # =========================================================================
    elif nav_selection == "⚖️ Model Comparison":
        st.title("⚖️ Side-by-Side Model Comparison")
        st.write(
            "Inspect how each individual algorithm behaves on the exact same user profile or seed game. "
            "Switch between models to observe ranking divergence, popularity bias, and thematic alignment."
        )

        comp_tabs = st.tabs([
            "🔥 Popularity Baseline",
            "📄 Content-Based Filtering",
            "🤝 Item-Item Collaborative",
            "📉 Matrix Factorization (SVD)",
            "⚡ Hybrid Recommendation",
        ])

        # Get active library
        library_items = [
            {"app_id": aid, "weight": 1.0}
            for aid, meta in st.session_state.user_library.items()
            if meta["status"] in ["liked", "played"]
        ]
        active_seed_ids = [item["app_id"] for item in library_items] if library_items else [292030]  # Default Witcher 3
        excluded_ids = set(st.session_state.user_library.keys())

        profile_label = (
            f"Active Library ({len(library_items)} games)"
            if library_items
            else "Default Seed (The Witcher 3: Wild Hunt)"
        )
        st.info(f"Comparing algorithms evaluated on: **{profile_label}**")

        # Tab 1: Popularity
        with comp_tabs[0]:
            st.markdown("### Model A: Bayesian Popularity Baseline")
            st.caption("Ranks by Bayesian-adjusted score $W = \\frac{v}{v+m}R + \\frac{m}{v+m}C$ scaled by review volume.")
            pop_recs = models["popularity"].recommend(k=10, exclude_app_ids=excluded_ids)
            for item in pop_recs:
                render_game_card(item, rank=item["rank"], show_actions=False)

        # Tab 2: Content-Based
        with comp_tabs[1]:
            st.markdown("### Model B: Multi-Modal Content-Based Filtering")
            st.caption("Sparse TF-IDF cosine similarity across community tags (60%), description (25%), and titles (15%).")
            if library_items:
                cb_recs = models["content"].recommend_user_profile(
                    liked_games=library_items, k=10, exclude_seen=excluded_ids
                )
            else:
                cb_recs = models["content"].recommend_similar(
                    app_id=active_seed_ids[0], k=10, exclude_seen=excluded_ids
                )
            for item in cb_recs:
                render_game_card(item, rank=item["rank"], show_actions=False)

        # Tab 3: Item-Item CF
        with comp_tabs[2]:
            st.markdown("### Model C: Item-Item Collaborative Filtering")
            st.caption("Co-play co-occurrence with Bayesian similarity shrinkage (λ=15.0) on implicit playtime confidence.")
            cf_recs = models["collaborative"].recommend_items(
                history_app_ids=active_seed_ids, k=10, exclude_seen=excluded_ids
            )
            if not cf_recs:
                st.warning("Collaborative model found no co-occurrence overlaps for this selection.")
            else:
                for item in cf_recs:
                    render_game_card(item, rank=item["rank"], show_actions=False)

        # Tab 4: Matrix Factorization
        with comp_tabs[3]:
            st.markdown("### Model D: Latent Factor Matrix Factorization (SVD)")
            st.caption("TruncatedSVD latent projections (d=64) on confidence-weighted implicit user-item interactions.")
            if models["svd"] is not None:
                valid_svd_aids = [a for a in active_seed_ids if a in models["svd"].app_to_idx]
                if valid_svd_aids:
                    # Projection
                    indices = [models["svd"].app_to_idx[a] for a in valid_svd_aids]
                    user_latent = models["svd"].item_factors[indices].mean(axis=0)
                    user_latent_norm = user_latent / (np.linalg.norm(user_latent) + 1e-8)
                    svd_scores = models["svd"].item_factors @ user_latent_norm
                    
                    for aid in excluded_ids:
                        if aid in models["svd"].app_to_idx:
                            svd_scores[models["svd"].app_to_idx[aid]] = -1.0
                    
                    top_idx = np.argsort(svd_scores)[::-1][:10]
                    svd_recs = []
                    for rank, idx in enumerate(top_idx, start=1):
                        sc = float(svd_scores[idx])
                        if sc <= 0:
                            break
                        aid = int(models["svd"].idx_to_app[idx])
                        svd_recs.append({
                            "app_id": aid,
                            "title": models["svd"].catalog_titles.get(aid, f"Game #{aid}"),
                            "rank": rank,
                            "score": round(sc, 4),
                            "explanation": f"High latent vector projection affinity ({sc:.3f}) in 64-dimensional feature space.",
                        })
                    for item in svd_recs:
                        render_game_card(item, rank=item["rank"], show_actions=False)
                else:
                    st.info("Selected games not present in active benchmark matrix.")
            else:
                st.info("Matrix Factorization model artifact not loaded.")

        # Tab 5: Hybrid
        with comp_tabs[4]:
            st.markdown("### Model E: Calibrated Hybrid Engine")
            st.caption("Weighted linear blend of min-max calibrated Content & Collaborative scores with MMR diversity.")
            hyb_recs = models["hybrid"].recommend_interactive(
                seed_games=library_items if library_items else active_seed_ids,
                k=10,
                alpha=0.5,
                exclude_seen=excluded_ids,
                use_diversity=True,
            )
            for item in hyb_recs:
                render_game_card(item, rank=item["rank"], show_actions=False)

    # =========================================================================
    # PAGE 5: OFFLINE EVALUATION & METRICS
    # =========================================================================
    elif nav_selection == "📊 Offline Evaluation":
        st.title("📊 Offline Ranking Evaluation & Metrics")
        st.write(
            "Empirical offline evaluation results measured on the held-out temporal evaluation dataset "
            "(`test_interactions.parquet`) strictly adhering to zero future-to-past data leakage."
        )

        if eval_metrics is None:
            st.warning(
                "Offline evaluation metrics have not yet been generated. "
                "Run `python3 scripts/evaluate_models.py` to produce reproducible offline metrics."
            )
        else:
            metrics_dict = eval_metrics.get("metrics", {})
            sample_size = eval_metrics.get("eval_sample_size", "N/A")
            eval_time = eval_metrics.get("timestamp", "N/A")

            st.caption(f"Evaluated on **{sample_size:,}** active test users (Leave-last-out) | Generated: `{eval_time}`")

            # Metrics Table
            st.subheader("🏆 Model Leaderboard (Held-Out Test Set)")
            rows = []
            for m_name, m_vals in metrics_dict.items():
                rows.append({
                    "Model": m_name,
                    "Hit Rate@5 (%)": round(m_vals.get("hit@5", 0.0) * 100, 2),
                    "Hit Rate@10 (%)": round(m_vals.get("hit@10", 0.0) * 100, 2),
                    "Hit Rate@20 (%)": round(m_vals.get("hit@20", 0.0) * 100, 2),
                    "NDCG@10": round(m_vals.get("ndcg@10", 0.0), 4),
                    "NDCG@20": round(m_vals.get("ndcg@20", 0.0), 4),
                    "Precision@10": round(m_vals.get("precision@10", 0.0), 4),
                    "MAP@10": round(m_vals.get("map@10", 0.0), 4),
                    "MRR": round(m_vals.get("mrr", 0.0), 4),
                })

            df_metrics = pd.DataFrame(rows).sort_values("Hit Rate@10 (%)", ascending=False)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # Interactive Bar Chart
            st.subheader("📈 Hit Rate@K Comparison (%)")
            chart_data = df_metrics.set_index("Model")[["Hit Rate@5 (%)", "Hit Rate@10 (%)", "Hit Rate@20 (%)"]]
            st.bar_chart(chart_data)

            # Analytical Takeaways
            st.markdown("---")
            st.subheader("🔍 Analytical Takeaways & Justifications")
            c_a1, c_a2 = st.columns(2)
            with c_a1:
                st.markdown(
                    """
                    **1. Collaborative Filtering Dominance**:
                    - Item-Item Collaborative Filtering achieved the highest Hit Rate@10 on active users.
                    - Co-play co-occurrence captures latent community gameplay affinities that raw metadata cannot capture.
                    
                    **2. Content-Based Value for Niche Items**:
                    - While Content-Based TF-IDF has lower hit rate on broad blockbusters, it solves the cold-start item problem for the 26% of games with zero reviews.
                    """
                )
            with c_a2:
                st.markdown(
                    """
                    **3. Hybrid Regularization & Balance**:
                    - The Hybrid model ($\alpha=0.5$) retains high hit rates while expanding catalog exploration and eliminating popularity echo chambers.
                    
                    **4. Zero-Leakage Temporal Integrity**:
                    - Splitting chronologically by leave-last-out ensures models only predict future game interactions from prior play history.
                    """
                )


if __name__ == "__main__":
    main()

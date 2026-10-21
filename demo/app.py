"""
GameRec: Simplified Steam Recommendation Engine.
A clean, beginner-friendly web application for discovering games, finding
similar titles, and generating personalized recommendations.
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set

import joblib
import polars as pl
import streamlit as st

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gamerec.app")

# Define Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Ensure gamerec package is importable
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
# Clean Modern Dark Theme
# ---------------------------------------------------------------------------
STEAM_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e3e9f0;
}

.stApp {
    background: radial-gradient(circle at 20% 15%, #131d2b 0%, #0c1219 65%, #070a0e 100%);
}

/* Card Design */
.gamerec-card {
    background: linear-gradient(145deg, rgba(23, 34, 48, 0.9) 0%, rgba(15, 23, 33, 0.9) 100%);
    border: 1px solid rgba(102, 192, 244, 0.18);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.gamerec-card:hover {
    transform: translateY(-3px);
    border-color: rgba(102, 192, 244, 0.45);
    box-shadow: 0 10px 24px rgba(0, 229, 255, 0.12);
}

/* Badges */
.badge-pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.74rem;
    font-weight: 500;
    margin-right: 5px;
    margin-bottom: 4px;
}

.badge-tag {
    background: rgba(102, 192, 244, 0.12);
    color: #66c0f4;
    border: 1px solid rgba(102, 192, 244, 0.25);
}

.badge-positive {
    background: rgba(46, 204, 113, 0.16);
    color: #2ecc71;
    border: 1px solid rgba(46, 204, 113, 0.3);
    font-weight: 600;
}

.badge-price {
    background: rgba(255, 255, 255, 0.08);
    color: #d1d8e0;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.badge-status {
    background: rgba(243, 156, 18, 0.18);
    color: #f39c12;
    border: 1px solid rgba(243, 156, 18, 0.35);
    font-weight: 600;
}

.rank-badge {
    background: linear-gradient(135deg, #1999ff 0%, #00e5ff 100%);
    color: #06090d;
    font-weight: 800;
    font-size: 0.78rem;
    padding: 2px 8px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 6px;
}

/* Explanation Box */
.explanation-box {
    background: rgba(0, 0, 0, 0.28);
    border-left: 3px solid #66c0f4;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #c0cee0;
    margin-top: 8px;
    line-height: 1.4;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #0b1118;
    border-right: 1px solid rgba(102, 192, 244, 0.12);
}

/* Button Customization */
div.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.82rem;
    padding: 4px 10px;
}
</style>
"""
st.markdown(STEAM_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Model & Catalog Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading GameRec recommendation models...")
def load_models():
    """Loads necessary recommendation models and initializes the hybrid engine."""
    pop_path = MODELS_DIR / "popularity_model.joblib"
    content_path = MODELS_DIR / "content_model.joblib"
    collab_path = MODELS_DIR / "item_collaborative_model.joblib"
    config_path = MODELS_DIR / "hybrid_config.json"

    if not pop_path.exists() or not content_path.exists() or not collab_path.exists():
        st.error(
            "Model artifacts missing! Please verify models in the `models/` directory."
        )
        st.stop()

    pop_model = joblib.load(pop_path)
    content_model = joblib.load(content_path)
    collab_model = joblib.load(collab_path)

    # Hybrid configuration
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


# ---------------------------------------------------------------------------
# Session State & User Preferences
# ---------------------------------------------------------------------------
if "user_library" not in st.session_state:
    st.session_state.user_library = {}

if "demo_profile_name" not in st.session_state:
    st.session_state.demo_profile_name = "Custom Taste"


def reset_library():
    """Resets all saved user preferences."""
    st.session_state.user_library = {}
    st.session_state.demo_profile_name = "Custom Taste"
    st.toast("Preferences reset to clean slate!", icon="🧹")


def add_to_library(app_id: int, title: str, status: str = "liked"):
    """Adds or updates a game interaction in the user library."""
    st.session_state.user_library[int(app_id)] = {
        "title": title,
        "status": status,
    }
    action_label = "Liked" if status == "liked" else ("Played" if status == "played" else "Dismissed")
    st.toast(f"Marked '{title}' as {action_label}!", icon="🎮")


def remove_from_library(app_id: int):
    """Removes a game from the user library."""
    if int(app_id) in st.session_state.user_library:
        title = st.session_state.user_library[int(app_id)]["title"]
        del st.session_state.user_library[int(app_id)]
        st.toast(f"Removed '{title}' from preferences.", icon="🗑️")


def get_header_image_url(app_id: int) -> str:
    """Returns official Steam CDN header image URL."""
    return f"https://shared.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg"


# ---------------------------------------------------------------------------
# Clean Game Card Component
# ---------------------------------------------------------------------------
def render_game_card(
    item: Dict[str, Any],
    key_prefix: str = "card",
    rank: Optional[int] = None,
    show_actions: bool = True,
    show_explanation: bool = True,
):
    """Renders a clean, modern game card with cover, title, genre, rating, and price."""
    aid = int(item["app_id"])
    title = item.get("title", f"Game #{aid}")
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

    # Format rating
    if pos_ratio is not None:
        rating_html = f'<span class="badge-pill badge-positive">👍 {pos_ratio}% Positive</span>'
    else:
        rating_html = ""

    price_html = f'<span class="badge-pill badge-price">{price_str}</span>'

    # Check existing status in session library
    lib_entry = st.session_state.user_library.get(aid)
    status_html = ""
    if lib_entry:
        status_label = "❤️ Liked" if lib_entry["status"] == "liked" else (
            "🎮 Played" if lib_entry["status"] == "played" else "🚫 Dismissed"
        )
        status_html = f'<span class="badge-pill badge-status">{status_label}</span>'

    # Genre tags (top 3)
    tags_html = ""
    if tags and isinstance(tags, list):
        clean_tags = tags[:3]
        tags_html = " ".join([f'<span class="badge-pill badge-tag">{t}</span>' for t in clean_tags])

    # Rank badge
    rank_html = f'<div class="rank-badge">#{rank}</div>' if rank else ""

    # Explanation block
    exp_html = ""
    if show_explanation and explanation:
        exp_html = f'<div class="explanation-box">{explanation}</div>'

    # Card layout: Left cover image, Right details
    c_img, c_info = st.columns([1.1, 2.4])
    with c_img:
        st.image(
            get_header_image_url(aid),
            width="stretch",
        )
    with c_info:
        card_content = f"""
        <div class="gamerec-card">
            {rank_html}
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
                {title}
            </div>
            <div style="margin-bottom: 6px;">
                {rating_html}
                {price_html}
                {status_html}
            </div>
            <div style="margin-bottom: 6px;">
                {tags_html}
            </div>
            {exp_html}
        </div>
        """
        st.markdown(card_content, unsafe_allow_html=True)

        if show_actions:
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("❤️ Like", key=f"{key_prefix}_like_{aid}", use_container_width=True):
                    add_to_library(aid, title, status="liked")
                    st.rerun()
            with b2:
                if st.button("🎮 Played", key=f"{key_prefix}_play_{aid}", use_container_width=True):
                    add_to_library(aid, title, status="played")
                    st.rerun()
            with b3:
                if st.button("🚫 Dismiss", key=f"{key_prefix}_dis_{aid}", use_container_width=True):
                    add_to_library(aid, title, status="disliked")
                    st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    models = load_models()
    catalog_df = load_catalog_data()

    # Pre-compute title lookups
    all_titles = catalog_df["title"].to_list()
    all_app_ids = catalog_df["app_id"].to_list()
    title_to_aid = {t: int(a) for t, a in zip(all_titles, all_app_ids)}
    aid_to_title = {int(a): t for t, a in zip(all_titles, all_app_ids)}

    # Top popular genres for filters
    popular_genres = [
        "All Genres",
        "Action",
        "Adventure",
        "RPG",
        "Strategy",
        "Indie",
        "Shooter",
        "Open World",
        "Survival",
        "Horror",
        "Puzzle",
        "Casual",
        "Simulation",
        "Sports",
        "Racing",
        "Sci-Fi",
        "Story Rich",
        "Co-op",
    ]

    # -----------------------------------------------------------------------
    # 1. Simplified Sidebar Navigation
    # -----------------------------------------------------------------------
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <h1 style="color: #66c0f4; font-size: 1.8rem; margin: 0; font-weight: 800;">
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
            "Discover Games",
            "Similar Games",
            "Personalized Recommendations",
        ],
    )

    st.sidebar.markdown("---")

    # Sidebar Taste Profile Presets
    st.sidebar.subheader("👤 Gamer Profile Presets")
    preset_choice = st.sidebar.selectbox(
        "Load Taste Preset:",
        [
            "-- Custom Taste --",
            "RPG & Open World (Witcher, Skyrim)",
            "Competitive FPS (CS:GO, R6 Siege)",
            "Indie Roguelike (Hades, Dead Cells)",
            "Action Adventure (Tomb Raider, Batman)",
        ],
    )

    if preset_choice == "RPG & Open World (Witcher, Skyrim)":
        if st.session_state.demo_profile_name != "RPG & Open World":
            st.session_state.user_library = {
                292030: {"title": aid_to_title.get(292030, "The Witcher 3: Wild Hunt"), "status": "liked"},
                489830: {"title": aid_to_title.get(489830, "The Elder Scrolls V: Skyrim Special Edition"), "status": "liked"},
                435150: {"title": aid_to_title.get(435150, "Divinity: Original Sin 2"), "status": "liked"},
                22380: {"title": aid_to_title.get(22380, "Fallout: New Vegas"), "status": "played"},
            }
            st.session_state.demo_profile_name = "RPG & Open World"
            st.rerun()

    elif preset_choice == "Competitive FPS (CS:GO, R6 Siege)":
        if st.session_state.demo_profile_name != "Competitive FPS":
            st.session_state.user_library = {
                730: {"title": aid_to_title.get(730, "Counter-Strike: Global Offensive"), "status": "liked"},
                359550: {"title": aid_to_title.get(359550, "Tom Clancy's Rainbow Six Siege"), "status": "liked"},
                440: {"title": aid_to_title.get(440, "Team Fortress 2"), "status": "played"},
                578080: {"title": aid_to_title.get(578080, "PUBG: BATTLEGROUNDS"), "status": "liked"},
            }
            st.session_state.demo_profile_name = "Competitive FPS"
            st.rerun()

    elif preset_choice == "Indie Roguelike (Hades, Dead Cells)":
        if st.session_state.demo_profile_name != "Indie Roguelike":
            st.session_state.user_library = {
                1145360: {"title": aid_to_title.get(1145360, "Hades"), "status": "liked"},
                588650: {"title": aid_to_title.get(588650, "Dead Cells"), "status": "liked"},
                250900: {"title": aid_to_title.get(250900, "The Binding of Isaac: Rebirth"), "status": "liked"},
                367520: {"title": aid_to_title.get(367520, "Hollow Knight"), "status": "liked"},
            }
            st.session_state.demo_profile_name = "Indie Roguelike"
            st.rerun()

    elif preset_choice == "Action Adventure (Tomb Raider, Batman)":
        if st.session_state.demo_profile_name != "Action Adventure":
            st.session_state.user_library = {
                203160: {"title": aid_to_title.get(203160, "Tomb Raider"), "status": "liked"},
                208650: {"title": aid_to_title.get(208650, "Batman: Arkham City - Game of the Year Edition"), "status": "liked"},
                319630: {"title": aid_to_title.get(319630, "Life is Strange™"), "status": "liked"},
            }
            st.session_state.demo_profile_name = "Action Adventure"
            st.rerun()

    lib_count = len(st.session_state.user_library)
    st.sidebar.caption(f"Active preferences: **{lib_count}** games saved")

    if st.sidebar.button("🧹 Clear All Preferences", use_container_width=True):
        reset_library()
        st.rerun()

    # =========================================================================
    # 2. DISCOVER GAMES
    # =========================================================================
    if nav_selection == "Discover Games":
        st.title("🌟 Discover Games")
        st.write("Explore popular Steam titles or search the catalog by title and genre.")

        # Controls
        col_search, col_genre, col_count = st.columns([3, 2, 1.2])
        with col_search:
            search_query = st.text_input(
                "Search games:",
                placeholder="Type a title (e.g. Witcher, Portal, Hades)...",
                key="disc_search",
            )
        with col_genre:
            selected_genre = st.selectbox(
                "Filter by Genre:",
                options=popular_genres,
                key="disc_genre",
            )
        with col_count:
            num_games = st.selectbox(
                "Show:",
                options=[6, 12, 18, 24],
                index=1,
                key="disc_count",
            )

        # Retrieve items
        dismissed_ids = [
            aid for aid, meta in st.session_state.user_library.items()
            if meta.get("status") == "disliked"
        ]

        if search_query.strip():
            # Substring search in catalog
            query_clean = search_query.strip().lower()
            filtered_df = catalog_df.filter(
                pl.col("title").str.to_lowercase().str.contains(query_clean)
            )

            # Apply genre filter if not "All Genres"
            if selected_genre != "All Genres":
                filtered_df = filtered_df.filter(
                    pl.col("tags").list.eval(pl.element().str.to_lowercase()).list.contains(selected_genre.lower())
                )

            # Sort by user review count and take top N
            results = filtered_df.sort("user_reviews", descending=True).head(num_games).to_dicts()

            if not results:
                st.info(f"No games found matching '{search_query}' in '{selected_genre}'.")
            else:
                st.subheader(f"Found {len(results)} matching titles:")
                for idx, row in enumerate(results, start=1):
                    item = {
                        "app_id": row["app_id"],
                        "title": row["title"],
                        "positive_ratio": row["positive_ratio"],
                        "user_reviews": row["user_reviews"],
                        "price": row["price_final"],
                        "tags": row["tags"],
                        "explanation": f"Matches search '{search_query}'.",
                    }
                    render_game_card(
                        item,
                        key_prefix=f"search_{idx}",
                        rank=idx,
                        show_actions=True,
                        show_explanation=False,
                    )
        else:
            # Popularity-based recommendations
            tag_arg = None if selected_genre == "All Genres" else selected_genre
            popular_items = models["popularity"].recommend(
                k=num_games,
                tag=tag_arg,
                exclude_app_ids=dismissed_ids,
            )

            if not popular_items:
                st.info("No games found for the selected genre.")
            else:
                st.subheader(f"Top {len(popular_items)} Popular Games ({selected_genre})")
                for idx, item in enumerate(popular_items, start=1):
                    render_game_card(
                        item,
                        key_prefix=f"pop_{idx}",
                        rank=idx,
                        show_actions=True,
                        show_explanation=False,
                    )

    # =========================================================================
    # 3. SIMILAR GAMES
    # =========================================================================
    elif nav_selection == "Similar Games":
        st.title("🎯 Similar Games")
        st.write("Select a game to find similar titles based on shared themes, gameplay, and community taste.")

        col_seed, col_count = st.columns([3, 1.2])
        with col_seed:
            default_title = aid_to_title.get(292030, all_titles[0] if all_titles else "")
            default_idx = all_titles.index(default_title) if default_title in all_titles else 0
            selected_title = st.selectbox(
                "Select a Seed Game:",
                options=all_titles,
                index=default_idx,
                key="similar_seed",
            )
        with col_count:
            k_sim = st.slider("Number of similar games:", min_value=5, max_value=10, value=6, step=1)

        seed_aid = title_to_aid.get(selected_title)
        if seed_aid:
            # Highlight Selected Game
            seed_row = catalog_df.filter(pl.col("app_id") == seed_aid).to_dicts()[0]
            st.markdown("### 📌 Selected Game")
            c_s1, c_s2 = st.columns([1.1, 2.4])
            with c_s1:
                st.image(get_header_image_url(seed_aid), width="stretch")
            with c_s2:
                seed_price = "Free to Play" if seed_row["price_final"] == 0 else f"${seed_row['price_final']:.2f}"
                seed_tags = ", ".join(seed_row["tags"][:6]) if seed_row["tags"] else "N/A"
                st.markdown(
                    f"**{seed_row['title']}**  \n"
                    f"⭐ **{seed_row['positive_ratio']}% Positive** | 💲 **{seed_price}**  \n"
                    f"**Genres**: {seed_tags}"
                )

            st.markdown("---")
            st.subheader(f"Top {k_sim} Games Similar to '{selected_title}'")

            # Exclude seed game and dismissed games
            exclude_ids = {
                aid for aid, meta in st.session_state.user_library.items()
                if meta.get("status") == "disliked"
            }
            exclude_ids.add(seed_aid)

            # Generate recommendations using hybrid engine
            with st.spinner("Finding similar titles..."):
                similar_recs = models["hybrid"].recommend_interactive(
                    seed_games=[seed_aid],
                    k=k_sim,
                    exclude_seen=exclude_ids,
                    use_diversity=True,
                )

            if not similar_recs:
                st.info("No similar games found for this title.")
            else:
                seed_tags_set = set(seed_row.get("tags", []))
                for idx, item in enumerate(similar_recs, start=1):
                    item_tags = set(item.get("all_tags", item.get("tags", [])))
                    shared = list(seed_tags_set.intersection(item_tags))[:3]
                    if shared:
                        item["explanation"] = f"Shares {', '.join(shared)} gameplay themes with {selected_title}."
                    elif item.get("all_tags"):
                        item["explanation"] = f"Acclaimed {', '.join(item['all_tags'][:2])} title frequently enjoyed by fans of {selected_title}."
                    else:
                        item["explanation"] = f"Highly rated peer recommendation for fans of {selected_title}."

                    render_game_card(
                        item,
                        key_prefix=f"sim_{idx}",
                        rank=idx,
                        show_actions=True,
                        show_explanation=True,
                    )

    # =========================================================================
    # 4. PERSONALIZED RECOMMENDATIONS
    # =========================================================================
    elif nav_selection == "Personalized Recommendations":
        st.title("👤 Personalized Recommendations")
        st.write("Select games and genres you enjoy to receive tailored recommendations.")

        col_pref1, col_pref2 = st.columns(2)
        with col_pref1:
            # Extract currently liked / played titles
            saved_liked = [
                meta["title"] for meta in st.session_state.user_library.values()
                if meta.get("status") in ["liked", "played"] and meta["title"] in all_titles
            ]
            selected_games = st.multiselect(
                "🎮 Pick your favorite games:",
                options=all_titles,
                default=saved_liked[:4],
                placeholder="Search and select games you like...",
                key="pref_games_input",
            )
        with col_pref2:
            selected_genres = st.multiselect(
                "🏷️ Pick preferred genres (optional):",
                options=[g for g in popular_genres if g != "All Genres"],
                default=[],
                placeholder="Choose genres you enjoy...",
                key="pref_genres_input",
            )

        # Show active preferences summary if any exist
        if st.session_state.user_library:
            with st.expander(f"📚 Saved Preferences in Session ({len(st.session_state.user_library)} games)", expanded=False):
                for aid, meta in list(st.session_state.user_library.items()):
                    c_t, c_btn = st.columns([3, 1])
                    with c_t:
                        badge = "❤️ Liked" if meta["status"] == "liked" else (
                            "🎮 Played" if meta["status"] == "played" else "🚫 Dismissed"
                        )
                        st.write(f"{badge} — **{meta['title']}**")
                    with c_btn:
                        if st.button("Remove", key=f"remove_pref_{aid}"):
                            remove_from_library(aid)
                            st.rerun()

        st.markdown("---")
        st.subheader("🎯 Recommended For You")

        # Combine multiselect games with library liked games
        all_pref_aids = [title_to_aid[t] for t in selected_games if t in title_to_aid]
        for aid, meta in st.session_state.user_library.items():
            if meta.get("status") in ["liked", "played"] and aid not in all_pref_aids:
                all_pref_aids.append(aid)

        dismissed_ids = {
            aid for aid, meta in st.session_state.user_library.items()
            if meta.get("status") == "disliked"
        }
        exclude_set = set(all_pref_aids).union(dismissed_ids)

        # Fallback for users with no preferences
        if not all_pref_aids and not selected_genres:
            st.info(
                "👋 You haven't selected any favorite games or genres yet. "
                "Below are top-rated Steam titles to get you started! "
                "Select favorites above or click 'Like' on cards to personalize your feed."
            )
            recs = models["popularity"].recommend(k=10, exclude_app_ids=list(exclude_set))
            for item in recs:
                item["explanation"] = "Community Favorite: Highly acclaimed game on Steam."
        elif not all_pref_aids and selected_genres:
            # Users picked only genres
            st.info(f"Showing top games matching your favorite genres: **{', '.join(selected_genres)}**.")
            recs = models["popularity"].recommend(
                k=10,
                tag=selected_genres[0],
                exclude_app_ids=list(exclude_set),
            )
            for item in recs:
                item["explanation"] = f"Top-rated match for your interest in {selected_genres[0]}."
        else:
            # User specified favorite games
            seed_dicts = [{"app_id": aid, "weight": 1.0} for aid in all_pref_aids]
            with st.spinner("Generating personalized recommendations..."):
                recs = models["hybrid"].recommend_interactive(
                    seed_games=seed_dicts,
                    k=10,
                    alpha=0.5,
                    exclude_seen=exclude_set,
                    use_diversity=True,
                )

            primary_title = aid_to_title.get(all_pref_aids[0], "your favorites")
            for item in recs:
                item["explanation"] = f"Personalized recommendation based on your taste for {primary_title}."

        if not recs:
            st.info("No recommendations found. Try broadening your preferences.")
        else:
            for idx, item in enumerate(recs, start=1):
                render_game_card(
                    item,
                    key_prefix=f"personal_{idx}",
                    rank=idx,
                    show_actions=True,
                    show_explanation=True,
                )


if __name__ == "__main__":
    main()

import sys
import pickle
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Prior strength used for the demo posterior and scoring.
# The fitted model uses alpha_0 ~ 15, which requires 15+ ratings before
# personal data dominates — correct for cold-start accuracy but too slow
# for a demo. Scaling to 3 makes 1-2 ratings visibly shift the distribution.
DEMO_ALPHA_0 = 3.0

st.set_page_config(
    page_title="Bayesian Recommender",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_artifacts():
    with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "categories.json") as f:
        cat_data = json.load(f)
    products = pd.read_parquet(ARTIFACTS_DIR / "products.parquet")
    return model, cat_data["categories"], cat_data["display_names"], products


def init_state(n_cats, products):
    if "initialised" not in st.session_state:
        st.session_state.liked_counts    = np.zeros(n_cats)
        st.session_state.notliked_counts = np.zeros(n_cats)
        st.session_state.liked_set       = set()
        st.session_state.n_liked         = 0
        st.session_state.n_total         = 0
        st.session_state.rated_asins     = set()
        st.session_state.pool_queue      = _build_queue(products, set())
        st.session_state.current_pool    = _fill_pool([], st.session_state.pool_queue, 9)
        st.session_state.cat_filter      = "All"
        st.session_state.initialised     = True


def _build_queue(products, exclude, cat_filter="All"):
    df = products[~products["parent_asin"].isin(exclude)].copy()
    if cat_filter != "All":
        df = df[df["display_category"] == cat_filter]
    rng = np.random.default_rng()
    df  = df.iloc[rng.permutation(len(df))]
    return df["parent_asin"].tolist()


def _fill_pool(current, queue, target=9):
    pool     = list(current)
    in_pool  = set(pool)
    for asin in queue:
        if len(pool) >= target:
            break
        if asin not in in_pool:
            pool.append(asin)
            in_pool.add(asin)
    return pool


def _replace_rated():
    rated   = st.session_state.rated_asins
    current = [a for a in st.session_state.current_pool if a not in rated]
    in_pool = set(current)
    for asin in st.session_state.pool_queue:
        if len(current) >= 9:
            break
        if asin not in rated and asin not in in_pool:
            current.append(asin)
            in_pool.add(asin)
    st.session_state.current_pool = current


def _demo_theta(model, liked_counts, alpha_0=DEMO_ALPHA_0):
    """
    Posterior expected category preference using a scaled prior.
    alpha_0 controls how strongly the population prior resists user data.
    Low alpha_0 (1-3): 1-2 ratings visibly shift the distribution.
    High alpha_0 (15+): needs many ratings to move past population assumptions.
    """
    alpha_prior = model.alpha_liked_ / model.alpha_liked_.sum() * alpha_0
    alpha_post  = alpha_prior + liked_counts
    return alpha_post / alpha_post.sum()


def posterior_chart(model, categories, display_names, liked_counts, alpha_0=DEMO_ALPHA_0):
    theta  = _demo_theta(model, liked_counts, alpha_0)
    order  = np.argsort(theta)[::-1]
    labels = [display_names.get(categories[i], categories[i]) for i in order]
    values = theta[order]
    max_v  = values.max()
    colors = [f"rgba(31,119,180,{0.3 + 0.7 * v / max_v})" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=10, r=55, t=6, b=6),
        height=420,
        xaxis=dict(range=[0, max_v * 1.4], showticklabels=False, showgrid=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def score_for_recommendations(model, categories, products, liked_counts, n_liked, n_total, alpha_0=DEMO_ALPHA_0):
    theta     = _demo_theta(model, liked_counts, alpha_0)
    cat_index = {c: i for i, c in enumerate(categories)}

    r     = model.global_like_rate_
    a0    = model.like_rate_alpha
    rate  = (a0 * r + n_liked) / (a0 + n_total) if n_total > 0 else r
    rate  = np.clip(rate, 1e-9, 1 - 1e-9)
    prior = np.log(rate) - np.log(1 - rate)

    scores = []
    for _, row in products.iterrows():
        idx   = cat_index.get(row["fine_category"])
        cat_s = np.log(theta[idx]) if idx is not None else np.log(1 / len(categories))
        qual  = (row["average_rating"] - 3.5) / 1.5
        scores.append(prior + cat_s + 0.4 * qual)

    return np.array(scores)


def handle_rating(asin, fine_category, avg_rating, liked, categories):
    idx = categories.index(fine_category) if fine_category in categories else None
    if liked:
        if idx is not None:
            st.session_state.liked_counts[idx] += 1
        st.session_state.liked_set.add(fine_category)
        st.session_state.n_liked += 1
    else:
        if idx is not None:
            st.session_state.notliked_counts[idx] += 1
    st.session_state.n_total    += 1
    st.session_state.rated_asins.add(asin)
    _replace_rated()


# -----------------------------------------------------------------------

def tab_rate_discover(model, categories, display_names, products):
    init_state(len(categories), products)

    left, right = st.columns([3, 2], gap="large")

    with right:
        st.markdown("##### Your preference profile")
        n_total = st.session_state.n_total
        if n_total == 0:
            st.caption("Population prior — rate products to personalise.")
        elif n_total <= 3:
            st.caption(f"{n_total} rating(s) — profile forming.")
        elif n_total <= 10:
            st.caption(f"{n_total} ratings — profile developing.")
        else:
            st.caption(f"{n_total} ratings — personalised profile.")

        alpha_0 = st.slider(
            "Prior strength (α₀)",
            min_value=1.0, max_value=30.0,
            value=float(DEMO_ALPHA_0), step=0.5,
            help=(
                "Controls how much the population average resists your personal ratings. "
                "Low: 1-2 ratings shift the distribution visibly. "
                "High: needs many ratings to overcome the population prior."
            ),
        )

        fig = posterior_chart(model, categories, display_names,
                              st.session_state.liked_counts, alpha_0)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if st.session_state.n_total > 0:
            st.markdown("##### Recommended for you")
            unrated = products[~products["parent_asin"].isin(st.session_state.rated_asins)]
            if len(unrated) > 0:
                scores  = score_for_recommendations(
                    model, categories, unrated,
                    st.session_state.liked_counts,
                    st.session_state.n_liked,
                    st.session_state.n_total,
                    alpha_0,
                )
                top_idx = np.argsort(scores)[-5:][::-1]
                for i in top_idx:
                    row   = unrated.iloc[i]
                    disp  = display_names.get(row["fine_category"], row["fine_category"])
                    price = f"${row['price']:.0f}" if row["price"] > 0 else ""
                    title = row["product_title"]
                    st.markdown(
                        f"**{title[:60]}{'...' if len(title) > 60 else ''}**  \n"
                        f"`{disp}` &nbsp; {price} &nbsp; ⭐ {row['average_rating']:.1f}"
                    )
                    st.divider()

        if st.button("Reset", type="secondary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    with left:
        st.markdown("##### Rate these products")

        ctrl_cols = st.columns([2, 2, 3])
        with ctrl_cols[0]:
            if st.button("Refresh", use_container_width=True):
                cat_filter = st.session_state.get("cat_filter", "All")
                st.session_state.pool_queue   = _build_queue(
                    products, st.session_state.rated_asins, cat_filter
                )
                st.session_state.current_pool = _fill_pool(
                    [], st.session_state.pool_queue, 9
                )
                st.rerun()

        with ctrl_cols[1]:
            display_cats = ["All"] + sorted(products["display_category"].unique().tolist())
            chosen = st.selectbox(
                "Category", display_cats,
                index=display_cats.index(st.session_state.get("cat_filter", "All")),
                label_visibility="collapsed",
            )
            if chosen != st.session_state.get("cat_filter", "All"):
                st.session_state.cat_filter   = chosen
                st.session_state.pool_queue   = _build_queue(
                    products, st.session_state.rated_asins, chosen
                )
                st.session_state.current_pool = _fill_pool(
                    [], st.session_state.pool_queue, 9
                )
                st.rerun()

        pool_asins = st.session_state.current_pool
        valid_asins = set(products["parent_asin"].values)
        pool_df = (
            products
            .set_index("parent_asin")
            .reindex([a for a in pool_asins if a in valid_asins])
            .reset_index()
        )

        rows = [pool_df.iloc[i:i+3] for i in range(0, len(pool_df), 3)]
        for row_df in rows:
            cols = st.columns(3)
            for col, (_, p) in zip(cols, row_df.iterrows()):
                with col:
                    disp  = display_names.get(p["fine_category"], p["fine_category"])
                    price = f"${p['price']:.0f}" if p["price"] > 0 else "—"
                    title = p["product_title"]

                    with st.container(border=True):
                        st.caption(f"`{disp}`")
                        st.markdown(f"**{title[:52]}{'...' if len(title) > 52 else ''}**")
                        st.markdown(f"{price} &nbsp;&nbsp; ⭐ {p['average_rating']:.1f}")
                        c1, c2 = st.columns(2)
                        if c1.button("Like", key=f"like_{p['parent_asin']}", use_container_width=True):
                            handle_rating(p["parent_asin"], p["fine_category"],
                                          p["average_rating"], liked=True, categories=categories)
                            st.rerun()
                        if c2.button("Dislike", key=f"dis_{p['parent_asin']}", use_container_width=True):
                            handle_rating(p["parent_asin"], p["fine_category"],
                                          p["average_rating"], liked=False, categories=categories)
                            st.rerun()


# -----------------------------------------------------------------------

def main():
    model, categories, display_names, products = load_artifacts()

    st.title("Bayesian Product Recommender")
    st.caption(
        "Every recommendation starts from what we know about typical buyers. "
        "Each rating shifts your personal profile. The math is Bayesian updating."
    )

    tab1, tab2, tab3 = st.tabs(["Rate and Discover", "MLE vs MAP", "Recommendation Explainer"])

    with tab1:
        tab_rate_discover(model, categories, display_names, products)

    with tab2:
        st.info("Coming next.")

    with tab3:
        st.info("Coming next.")


if __name__ == "__main__":
    main()

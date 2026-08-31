import io
import json
import random
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning

from backend.data_generator import generate_student_data
from backend.genetic_algorithm import GeneticAlgorithm

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------------------------------------
# Page Config & Custom Styling
# --------------------------------------------------
st.set_page_config(
    page_title="ElasticNet GA Optimizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Status Badges */
    .badge-selected {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-dropped {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Header Styling */
    .header-banner {
        background: linear-gradient(90deg, #1e1b4b 0%, #0f172a 100%);
        border-bottom: 2px solid #3b82f6;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .header-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_default_dataset():
    if "df" not in st.session_state:
        st.session_state.df = generate_student_data(num_students=500, noise_level=5, random_state=42)
        st.session_state.target_col = "Final_Marks"
        st.session_state.data_source = "synthetic"


get_default_dataset()

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-title">🧬 ElasticNet GA Optimizer</div>
    <div class="header-subtitle">
        Simultaneous Hyperparameter Tuning (α, l₁-ratio) & L₀ Feature Selection powered by Genetic Algorithms
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # 1. Dataset Selection
    st.subheader("1. Data Source")
    data_option = st.radio(
        "Choose Dataset Source",
        ["Synthetic Student Performance", "Upload Custom CSV"],
        key="data_source_radio"
    )

    if data_option == "Synthetic Student Performance":
        with st.expander("Synthetic Data Parameters", expanded=False):
            num_samples = st.slider("Sample Count", 100, 2000, 500, step=50)
            noise_val = st.slider("Noise Level (std)", 0.0, 15.0, 5.0, step=0.5)
            seed_val = st.number_input("Random Seed", value=42, step=1)

            if st.button("🔄 Regenerate Synthetic Data", use_container_width=True):
                st.session_state.df = generate_student_data(
                    num_students=num_samples,
                    noise_level=noise_val,
                    random_state=int(seed_val)
                )
                st.session_state.target_col = "Final_Marks"
                st.session_state.data_source = "synthetic"
                st.session_state.pop("ga_results", None)
                st.toast("Generated new synthetic dataset!", icon="✅")

    else:
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.session_state.df = uploaded_df
                st.session_state.data_source = "uploaded"
                st.session_state.pop("ga_results", None)
                st.success(f"Uploaded {uploaded_file.name} ({len(uploaded_df)} rows, {len(uploaded_df.columns)} cols)")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

    # Select Target Variable
    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found in the dataset!")
        st.stop()

    default_target_idx = (
        numeric_cols.index(st.session_state.target_col)
        if hasattr(st.session_state, "target_col") and st.session_state.target_col in numeric_cols
        else len(numeric_cols) - 1
    )

    target_column = st.selectbox("Target Variable (y)", numeric_cols, index=default_target_idx)
    st.session_state.target_col = target_column

    st.markdown("---")

    # 2. GA Hyperparameters
    st.subheader("2. Genetic Algorithm Setup")

    pop_size = st.slider("Population Size", min_value=10, max_value=100, value=30, step=5)
    num_generations = st.slider("Generations", min_value=5, max_value=100, value=30, step=5)

    col_ga1, col_ga2 = st.columns(2)
    with col_ga1:
        crossover_rate = st.slider("Crossover Rate", 0.1, 1.0, 0.8, step=0.05)
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.50, 0.10, step=0.01)
    with col_ga2:
        gamma_penalty = st.slider("Feature Penalty (γ)", 0.00, 0.20, 0.02, step=0.005, help="Penalty for each included feature to favor parsimony.")
        tournament_size = st.slider("Tournament Size", 2, 8, 3)

    elitism_count = st.slider("Elitism Count", 1, 5, 2)

    st.markdown("---")
    run_ga_btn = st.button("🚀 Start GA Optimization", type="primary", use_container_width=True)

# --------------------------------------------------
# Main Content Tabs
# --------------------------------------------------
tab_explorer, tab_ga, tab_benchmark, tab_export = st.tabs([
    "📊 Data Explorer",
    "🧬 Optimization & Evolution",
    "📈 Benchmark & Interpretability",
    "💾 Export & Report"
])

# ==================================================
# TAB 1: Data Explorer
# ==================================================
with tab_explorer:
    st.subheader("Dataset Overview")
    df = st.session_state.df
    target_col = st.session_state.target_col

    feature_cols = [c for c in df.columns if c != target_col]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Samples</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-sub">Rows in dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Features</div>
            <div class="metric-value">{len(feature_cols)}</div>
            <div class="metric-sub">Candidate predictors</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Target Variable</div>
            <div class="metric-value" style="font-size: 1.3rem; word-break: break-all;">{target_col}</div>
            <div class="metric-sub">Regression outcome</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        missing_count = df.isnull().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Missing Values</div>
            <div class="metric-value">{missing_count}</div>
            <div class="metric-sub">Null cells</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("#### 📋 Data Preview")
        st.dataframe(df.head(100), use_container_width=True, height=350)

    with col_right:
        st.markdown(f"#### 🎯 Feature Correlation with `{target_col}`")

        # Correlation computation
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col in numeric_df.columns:
            corrs = numeric_df.corr()[target_col].drop(target_col).sort_values(ascending=True)

            fig_corr = px.bar(
                x=corrs.values,
                y=corrs.index,
                orientation='h',
                color=corrs.values,
                color_continuous_scale="Viridis",
                labels={"x": "Pearson Correlation", "y": "Feature"},
                title=f"Correlation with {target_col}"
            )
            fig_corr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=350
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Target column is non-numeric.")

# ==================================================
# GA Optimization Logic Execution
# ==================================================
if run_ga_btn:
    st.session_state.active_tab = 1
    with tab_ga:
        st.subheader("⚡ Live Genetic Algorithm Evolution")

        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        history_table_placeholder = st.empty()

        # Initialize GA
        ga_instance = GeneticAlgorithm(
            dataframe=st.session_state.df,
            target_column=st.session_state.target_col,
            population_size=pop_size,
            generations=num_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            gamma=gamma_penalty,
            tournament_size=tournament_size,
            elitism_count=elitism_count
        )

        history_records = []
        gen_numbers = []
        best_fitnesses = []

        def live_progress_callback(gen, total_gen, best_fitness, best_info):
            pct = gen / total_gen
            progress_bar.progress(pct)

            gen_numbers.append(gen)
            best_fitnesses.append(best_fitness)

            status_placeholder.markdown(
                f"**Evolution Progress:** Generation `{gen}/{total_gen}` ({int(pct*100)}%)"
            )

            # Live Metrics
            with metrics_placeholder.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Best Fitness", f"{best_fitness:.4f}")
                m2.metric("Decoded Alpha (α)", f"{best_info['alpha']:.5f}")
                m3.metric("Decoded L1 Ratio", f"{best_info['l1_ratio']:.4f}")
                m4.metric("Selected Features", f"{best_info['num_selected']} / {best_info['total_features']}")

            # Live Chart
            fig_live = px.line(
                x=gen_numbers,
                y=best_fitnesses,
                labels={"x": "Generation", "y": "Best Fitness (CV R² - Penalty)"},
                title="Fitness Trajectory over Generations",
                markers=True
            )
            fig_live.update_traces(line_color="#38bdf8", line_width=3, marker=dict(size=6, color="#818cf8"))
            fig_live.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                height=380,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            chart_placeholder.plotly_chart(fig_live, use_container_width=True)

            history_records.append({
                "Generation": gen,
                "Best_Fitness": round(best_fitness, 4),
                "Alpha": round(best_info['alpha'], 5),
                "L1_Ratio": round(best_info['l1_ratio'], 4),
                "Num_Features": best_info['num_selected'],
                "Selected_Features": ", ".join(best_info['selected_features'])
            })

        # Run Evolution
        best_chromosome_info = ga_instance.evolve(progress_callback=live_progress_callback)

        # Store in session state
        st.session_state.ga_results = {
            "best_info": best_chromosome_info,
            "history": history_records,
            "ga_instance": ga_instance,
            "target_col": st.session_state.target_col
        }

        st.toast("🎉 Optimization Completed!", icon="🚀")

# ==================================================
# TAB 2: Optimization & Evolution View (Saved State)
# ==================================================
with tab_ga:
    if "ga_results" in st.session_state:
        results = st.session_state.ga_results
        best_info = results["best_info"]
        history_df = pd.DataFrame(results["history"])

        st.subheader("🏆 Best Solution Discovered")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Best Fitness Score", f"{best_info['fitness']:.4f}")
        b2.metric("Optimal Alpha (α)", f"{best_info['alpha']:.6f}")
        b3.metric("Optimal L1 Ratio", f"{best_info['l1_ratio']:.4f}")
        b4.metric("Selected Features", f"{best_info['num_selected']} / {best_info['total_features']}")

        st.markdown("---")

        col_hist_chart, col_hist_table = st.columns([1.2, 0.8])

        with col_hist_chart:
            st.markdown("#### 📉 Fitness Convergence")
            fig_hist = px.line(
                history_df,
                x="Generation",
                y="Best_Fitness",
                markers=True,
                title="Best Fitness vs Generation"
            )
            fig_hist.update_traces(line_color="#38bdf8", line_width=3, marker=dict(size=6, color="#a855f7"))
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                height=380,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_hist_table:
            st.markdown("#### 📜 Evolution History Log")
            st.dataframe(history_df[["Generation", "Best_Fitness", "Alpha", "L1_Ratio", "Num_Features"]], use_container_width=True, height=380)

    elif not run_ga_btn:
        st.info("👈 Click **Start GA Optimization** in the sidebar to run the Genetic Algorithm search!")

# ==================================================
# TAB 3: Benchmark & Interpretability
# ==================================================
with tab_benchmark:
    if "ga_results" in st.session_state:
        st.subheader("🥊 Model Performance Benchmark")
        results = st.session_state.ga_results
        best_info = results["best_info"]

        df = st.session_state.df
        target_col = st.session_state.target_col
        feature_names = [c for c in df.columns if c != target_col]

        X_full = df[feature_names]
        y_full = df[target_col]

        selected_feats = best_info["selected_features"]
        if not selected_feats:
            selected_feats = feature_names

        X_ga = df[selected_feats]

        # 80/20 Train Test Split for evaluation
        X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        X_train_ga = X_train_full[selected_feats]
        X_test_ga = X_test_full[selected_feats]

        # Benchmark Models
        models = {
            "🧬 GA-Optimized ElasticNet": ElasticNet(alpha=best_info["alpha"], l1_ratio=best_info["l1_ratio"], max_iter=5000, random_state=42),
            "⚖️ Default ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=42),
            "📏 OLS Linear Regression": LinearRegression(),
            "🎯 Lasso (L1)": Lasso(alpha=0.1, max_iter=5000, random_state=42),
            "🏔️ Ridge (L2)": Ridge(alpha=1.0, random_state=42),
        }

        benchmark_results = []
        ga_model_obj = None

        for name, model in models.items():
            if "GA-Optimized" in name:
                X_tr, X_te = X_train_ga, X_test_ga
                num_f = len(selected_feats)
            else:
                X_tr, X_te = X_train_full, X_test_full
                num_f = len(feature_names)

            model.fit(X_tr, y_train)
            preds = model.predict(X_te)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = calculate_rmse(y_test, preds)

            if "GA-Optimized" in name:
                ga_model_obj = model
                ga_preds = preds

            benchmark_results.append({
                "Model": name,
                "Features Used": f"{num_f} / {len(feature_names)}",
                "Test R² Score": round(r2, 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
            })

        bench_df = pd.DataFrame(benchmark_results)

        st.dataframe(
            bench_df.style.highlight_max(subset=["Test R² Score"], color="rgba(34, 197, 94, 0.4)")
                          .highlight_min(subset=["MAE", "RMSE"], color="rgba(34, 197, 94, 0.4)"),
            use_container_width=True
        )

        st.markdown("---")

        # Visualizations
        c_vis1, c_vis2 = st.columns(2)

        with c_vis1:
            st.markdown("#### 🧬 Feature Selection & Learned Weights")
            if ga_model_obj is not None:
                coef_df = pd.DataFrame({
                    "Feature": selected_feats,
                    "Coefficient Weight": ga_model_obj.coef_
                }).sort_values(by="Coefficient Weight", key=abs, ascending=True)

                fig_coef = px.bar(
                    coef_df,
                    x="Coefficient Weight",
                    y="Feature",
                    orientation="h",
                    color="Coefficient Weight",
                    color_continuous_scale="Blues",
                    title="Learned ElasticNet Coefficients"
                )
                fig_coef.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15, 23, 42, 0.6)",
                    font=dict(color="#f8fafc"),
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_coef, use_container_width=True)

        with c_vis2:
            st.markdown("#### 🎯 Actual vs. Predicted (Test Set)")
            eval_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": ga_preds,
                "Residual": y_test - ga_preds
            })

            fig_pred = px.scatter(
                eval_df,
                x="Actual",
                y="Predicted",
                hover_data=["Residual"],
                title="Actual vs Predicted Outcome"
            )
            # Perfect fit reference line
            min_val = min(eval_df["Actual"].min(), eval_df["Predicted"].min())
            max_val = max(eval_df["Actual"].max(), eval_df["Predicted"].max())
            fig_pred.add_shape(
                type="line", line=dict(dash="dash", color="#ef4444", width=2),
                x0=min_val, y0=min_val, x1=max_val, y1=max_val
            )
            fig_pred.update_traces(marker=dict(size=7, color="#38bdf8", opacity=0.8))
            fig_pred.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#f8fafc"),
                height=360,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("#### 🧩 Feature Mask (Selected vs Dropped)")
        mask_data = []
        for feat in feature_names:
            is_sel = feat in selected_feats
            mask_data.append({
                "Feature Name": feat,
                "Status": "Selected ✅" if is_sel else "Excluded ❌",
                "Included Bit": 1 if is_sel else 0
            })
        st.dataframe(pd.DataFrame(mask_data), use_container_width=True)

    else:
        st.info("👈 Run GA Optimization in the sidebar to compare model benchmarks!")

# ==================================================
# TAB 4: Export & Report
# ==================================================
with tab_export:
    st.subheader("💾 Export Results & Parameters")

    if "ga_results" in st.session_state:
        results = st.session_state.ga_results
        best_info = results["best_info"]

        export_dict = {
            "project": "ElasticNet GA Optimizer",
            "target_variable": results["target_col"],
            "optimal_hyperparameters": {
                "alpha": best_info["alpha"],
                "l1_ratio": best_info["l1_ratio"],
                "alpha_gene": best_info["alpha_gene"],
                "l1_ratio_gene": best_info["l1_ratio_gene"],
            },
            "fitness_score": best_info["fitness"],
            "selected_features": best_info["selected_features"],
            "feature_bits": best_info["feature_bits"],
            "num_selected": best_info["num_selected"],
            "total_features": best_info["total_features"]
        }

        st.json(export_dict)

        c_exp1, c_exp2 = st.columns(2)

        with c_exp1:
            json_str = json.dumps(export_dict, indent=4)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_str,
                file_name="elasticnet_ga_optimization.json",
                mime="application/json",
                use_container_width=True
            )

        with c_exp2:
            feat_csv = pd.DataFrame({"selected_features": best_info["selected_features"]}).to_csv(index=False)
            st.download_button(
                label="📥 Download Selected Features CSV",
                data=feat_csv,
                file_name="selected_features.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.info("👈 Run GA Optimization first to generate exportable reports!")

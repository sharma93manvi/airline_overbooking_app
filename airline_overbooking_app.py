import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import binom

st.set_page_config(
    page_title="Airline Overbooking Simulator",
    page_icon="✈️",
    layout="wide",
)

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("✈️ Airline Overbooking Simulation Platform")
st.markdown(
    "Monte Carlo simulation to find the **optimal number of tickets to sell** "
    "above available capacity in order to maximize expected profit."
)
st.markdown("---")

# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Parameters")

n_seats = st.sidebar.number_input(
    "Number of Seats Available", min_value=10, max_value=2000, value=100, step=1
)
seat_price = st.sidebar.number_input(
    "Seat Price ($)", min_value=1, max_value=50_000, value=300, step=10
)
voucher_cost = st.sidebar.number_input(
    "Overbooking Voucher Cost ($ / bumped passenger)",
    min_value=1, max_value=50_000, value=500, step=10,
)
no_show_pct = st.sidebar.slider(
    "Probability of No-Show (%)", min_value=0.0, max_value=60.0, value=10.0, step=0.5
)
no_show_prob = no_show_pct / 100.0
show_prob = 1.0 - no_show_prob

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
n_sims = st.sidebar.selectbox(
    "Monte Carlo Iterations", [10_000, 50_000, 100_000], index=0,
    help="More iterations → smoother curves, slower runtime."
)
max_overbook = st.sidebar.slider(
    "Max Overbooking Range to Explore", min_value=10, max_value=200, value=50,
    help="Upper bound of extra tickets sold beyond seat capacity."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Assumptions**\n"
    "- No refund for no-shows (airline keeps the fare)\n"
    "- Customers show up independently with the same probability\n"
    "- Bumped passengers each receive a fixed dollar voucher\n"
    "- Overbooking = tickets sold **above** seat capacity"
)

# ── Core Simulation ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_simulation(n_seats, seat_price, voucher_cost, show_prob, n_sims, max_overbook):
    rng = np.random.default_rng(42)
    records = []

    for overbook in range(0, max_overbook + 1):
        tickets = n_seats + overbook
        revenue = seat_price * tickets  # collected upfront, no refunds

        # Simulate how many passengers actually show up
        arrivals = rng.binomial(tickets, show_prob, size=n_sims)

        bumped = np.maximum(0, arrivals - n_seats)
        net = revenue - voucher_cost * bumped

        records.append(
            dict(
                overbook=overbook,
                tickets=tickets,
                mean_profit=net.mean(),
                std_profit=net.std(),
                p05=np.percentile(net, 5),
                p25=np.percentile(net, 25),
                p75=np.percentile(net, 75),
                p95=np.percentile(net, 95),
                mean_bumped=bumped.mean(),
                prob_bump=(bumped > 0).mean(),
            )
        )

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def analytical_profits(n_seats, seat_price, voucher_cost, show_prob, max_overbook):
    """Closed-form expected profit for each overbooking level."""
    records = []
    for overbook in range(0, max_overbook + 1):
        S = n_seats + overbook
        revenue = seat_price * S
        # E[cost] = voucher * sum_{k=N+1}^{S} (k-N) * P(X=k), X~Binom(S, show_prob)
        ks = np.arange(n_seats + 1, S + 1)
        expected_cost = (
            voucher_cost * np.sum((ks - n_seats) * binom.pmf(ks, S, show_prob))
            if len(ks) > 0
            else 0.0
        )
        records.append(dict(overbook=overbook, analytical_profit=revenue - expected_cost))
    return pd.DataFrame(records)


with st.spinner("Running simulation…"):
    df = run_simulation(n_seats, seat_price, voucher_cost, show_prob, n_sims, max_overbook)
    df_anal = analytical_profits(n_seats, seat_price, voucher_cost, show_prob, max_overbook)

# Merge analytical into main df
df = df.merge(df_anal, on="overbook")

# Optimal rows
opt_idx = df["mean_profit"].idxmax()
opt = df.loc[opt_idx]
opt_anal_idx = df["analytical_profit"].idxmax()
opt_anal = df.loc[opt_anal_idx]

baseline_profit = seat_price * n_seats  # sell exactly N seats, no one gets bumped, all show → worst-case? No – baseline is E[profit] at overbook=0
baseline_mc = df.loc[df["overbook"] == 0, "mean_profit"].values[0]
gain = opt["mean_profit"] - baseline_mc

# ── KPI Cards ──────────────────────────────────────────────────────────────────
st.subheader("Optimal Strategy")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Seats to Overbook",
    f"+{int(opt['overbook'])}",
    f"{int(opt['tickets'])} tickets total",
)
k2.metric(
    "Expected Profit (Optimal)",
    f"${opt['mean_profit']:,.0f}",
    f"+${gain:,.0f} vs. no overbook",
)
k3.metric("Baseline Profit (No Overbook)", f"${baseline_mc:,.0f}")
k4.metric(
    "Avg. Bumped at Optimal",
    f"{opt['mean_bumped']:.2f}",
    help="Average passengers bumped per flight at the optimal overbooking level",
)
k5.metric(
    "Bump Probability at Optimal",
    f"{opt['prob_bump']*100:.1f}%",
    help="Share of flights where at least one passenger is bumped",
)

st.markdown("---")

# ── Main Charts ────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

# Chart 1 – Expected Profit Curve
with col_a:
    fig1 = go.Figure()

    # Confidence band (5th–95th percentile)
    fig1.add_trace(
        go.Scatter(
            x=pd.concat([df["overbook"], df["overbook"][::-1]]),
            y=pd.concat([df["p95"], df["p05"][::-1]]),
            fill="toself",
            fillcolor="rgba(65,105,225,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5th–95th Percentile",
            hoverinfo="skip",
        )
    )
    # IQR band
    fig1.add_trace(
        go.Scatter(
            x=pd.concat([df["overbook"], df["overbook"][::-1]]),
            y=pd.concat([df["p75"], df["p25"][::-1]]),
            fill="toself",
            fillcolor="rgba(65,105,225,0.20)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25th–75th Percentile",
            hoverinfo="skip",
        )
    )
    # Monte Carlo mean
    fig1.add_trace(
        go.Scatter(
            x=df["overbook"],
            y=df["mean_profit"],
            mode="lines+markers",
            name="MC Expected Profit",
            line=dict(color="royalblue", width=2.5),
            marker=dict(size=4),
        )
    )
    # Analytical line
    fig1.add_trace(
        go.Scatter(
            x=df["overbook"],
            y=df["analytical_profit"],
            mode="lines",
            name="Analytical Expected Profit",
            line=dict(color="darkgreen", width=2, dash="dot"),
        )
    )
    # Optimal marker
    fig1.add_vline(
        x=int(opt["overbook"]),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal: +{int(opt['overbook'])}",
        annotation_position="top right",
        annotation_font_color="red",
    )
    fig1.update_layout(
        title="Expected Profit vs. Overbooking Amount",
        xaxis_title="Extra Tickets Sold (Overbooking)",
        yaxis_title="Expected Profit ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2 – Bumping Statistics
with col_b:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=df["overbook"],
            y=df["mean_bumped"],
            name="Avg. Bumped Passengers",
            marker_color="salmon",
            opacity=0.75,
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=df["overbook"],
            y=df["prob_bump"] * 100,
            mode="lines",
            name="Prob. of Any Bump (%)",
            yaxis="y2",
            line=dict(color="darkorange", width=2.5),
        )
    )
    fig2.add_vline(
        x=int(opt["overbook"]),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal: +{int(opt['overbook'])}",
        annotation_position="top right",
        annotation_font_color="red",
    )
    fig2.update_layout(
        title="Bumping Risk vs. Overbooking Amount",
        xaxis_title="Extra Tickets Sold (Overbooking)",
        yaxis=dict(title="Avg. Bumped Passengers"),
        yaxis2=dict(
            title="Prob. of Any Bump (%)",
            overlaying="y",
            side="right",
            range=[0, 105],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Profit Distribution at Optimal ─────────────────────────────────────────────
st.subheader(f"Profit Distribution at Optimal Overbooking Level (+{int(opt['overbook'])} seats)")


@st.cache_data(show_spinner=False)
def get_profit_distribution(n_seats, seat_price, voucher_cost, show_prob, n_sims, overbook):
    rng = np.random.default_rng(99)
    tickets = n_seats + overbook
    arrivals = rng.binomial(tickets, show_prob, size=n_sims)
    bumped = np.maximum(0, arrivals - n_seats)
    return seat_price * tickets - voucher_cost * bumped


dist = get_profit_distribution(
    n_seats, seat_price, voucher_cost, show_prob, n_sims, int(opt["overbook"])
)

col_d1, col_d2 = st.columns([2, 1])
with col_d1:
    fig3 = px.histogram(
        dist,
        nbins=60,
        color_discrete_sequence=["steelblue"],
        labels={"value": "Profit ($)", "count": "Frequency"},
        title=f"Simulated Profit Distribution (n = {n_sims:,} flights)",
    )
    fig3.add_vline(
        x=float(np.mean(dist)),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean = ${np.mean(dist):,.0f}",
    )
    fig3.add_vline(
        x=float(np.percentile(dist, 5)),
        line_dash="dot",
        line_color="orange",
        annotation_text=f"5th %ile = ${np.percentile(dist, 5):,.0f}",
    )
    fig3.update_layout(showlegend=False, xaxis_title="Profit ($)", yaxis_title="Frequency")
    st.plotly_chart(fig3, use_container_width=True)

with col_d2:
    st.markdown("#### Summary Statistics")
    stats = {
        "Tickets Sold": f"{int(opt['tickets'])}",
        "Mean Profit": f"${np.mean(dist):,.0f}",
        "Std Deviation": f"${np.std(dist):,.0f}",
        "5th Percentile": f"${np.percentile(dist, 5):,.0f}",
        "25th Percentile": f"${np.percentile(dist, 25):,.0f}",
        "Median": f"${np.median(dist):,.0f}",
        "75th Percentile": f"${np.percentile(dist, 75):,.0f}",
        "95th Percentile": f"${np.percentile(dist, 95):,.0f}",
        "Avg. Bumped": f"{opt['mean_bumped']:.2f}",
        "Bump Probability": f"{opt['prob_bump']*100:.1f}%",
    }
    for label, val in stats.items():
        st.markdown(f"**{label}:** {val}")

st.markdown("---")

# ── Analytical Explanation ─────────────────────────────────────────────────────
with st.expander("Analytical Formula & Optimal Solution"):
    st.markdown("### Expected Profit Formula")
    st.latex(
        r"""
        E[\text{Profit}(S)] = p \cdot S \;-\; v \cdot \sum_{k=N+1}^{S}
        (k - N)\,\binom{S}{k}\,(1-q)^{\,k}\,q^{\,S-k}
        """
    )
    st.markdown(
        """
| Symbol | Meaning | Value |
|--------|---------|-------|
| $S$ | Tickets sold | optimized |
| $N$ | Seats available | {n_seats} |
| $p$ | Seat price | ${seat_price} |
| $v$ | Voucher cost per bumped passenger | ${voucher_cost} |
| $q$ | No-show probability | {no_show_pct:.1f}% |
| $1-q$ | Show-up probability | {show_pct:.1f}% |
        """.format(
            n_seats=n_seats,
            seat_price=seat_price,
            voucher_cost=voucher_cost,
            no_show_pct=no_show_pct * 100,
            show_pct=show_prob * 100,
        )
    )
    st.success(
        f"**Analytical Optimal:** Overbook by **{int(opt_anal['overbook'])}** extra seats "
        f"({int(opt_anal['tickets'])} tickets total) → "
        f"Expected Profit = **${opt_anal['analytical_profit']:,.2f}**"
    )

# ── Full Data Table ────────────────────────────────────────────────────────────
with st.expander("Full Simulation Results Table"):
    display = df[
        ["overbook", "tickets", "mean_profit", "std_profit", "p05", "p95",
         "mean_bumped", "prob_bump", "analytical_profit"]
    ].copy()
    display.columns = [
        "Overbook", "Tickets Sold", "MC Mean Profit",
        "Std Dev", "5th %ile", "95th %ile",
        "Avg Bumped", "Prob Bump", "Analytical Profit",
    ]
    for col in ["MC Mean Profit", "Std Dev", "5th %ile", "95th %ile", "Analytical Profit"]:
        display[col] = display[col].map("${:,.0f}".format)
    display["Avg Bumped"] = display["Avg Bumped"].map("{:.2f}".format)
    display["Prob Bump"] = display["Prob Bump"].map("{:.1%}".format)
    display["Overbook"] = display["Overbook"].map("+{}".format)

    st.dataframe(display, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "BAMS 503 · Airline Overbooking Simulation · "
    f"Monte Carlo with {n_sims:,} iterations per overbooking level"
)

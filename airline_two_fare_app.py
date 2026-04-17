import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Airline Revenue Management Simulator",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.title("Airline Revenue Management Simulator")
st.markdown(
    "Optimize **total tickets to sell** and the **leisure fare booking limit** "
    "to maximize expected profit, balancing overbooking risk against empty seats "
    "and fare-class revenue."
)
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Flight Parameters")

total_seats = st.sidebar.number_input(
    "Aircraft Capacity (seats)", min_value=50, max_value=500, value=150, step=1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("F1 — Business Fare")
st.sidebar.caption("Sold from 2 weeks before departure until flight departure")
f1_price = st.sidebar.number_input("F1 Ticket Price ($)", 100, 10000, 1500, 50)
f1_demand_mean = st.sidebar.number_input("F1 Demand — Mean", 1, 200, 20, 1)
f1_demand_std = st.sidebar.number_input("F1 Demand — Std Dev", 1, 50, 5, 1)
f1_noshow_pct = st.sidebar.slider("F1 No-Show Probability (%)", 0.0, 50.0, 15.0, 0.5)
st.sidebar.caption("Business no-shows receive a full refund (airline loses that revenue).")

st.sidebar.markdown("---")
st.sidebar.subheader("F2 — Leisure Fare")
st.sidebar.caption("Sold up until 2 weeks before departure")
f2_price = st.sidebar.number_input("F2 Ticket Price ($)", 50, 5000, 500, 50)
f2_demand_mean = st.sidebar.number_input("F2 Demand — Mean", 10, 500, 200, 5)
f2_demand_std = st.sidebar.number_input("F2 Demand — Std Dev", 1, 100, 20, 1)
f2_noshow_pct = st.sidebar.slider("F2 No-Show Probability (%)", 0.0, 50.0, 5.0, 0.5)
st.sidebar.caption("Leisure no-shows do NOT receive a refund (airline keeps the revenue).")

st.sidebar.markdown("---")
st.sidebar.subheader("Denied Boarding")
volunteer_prob_pct = st.sidebar.slider(
    "Volunteer Probability per Leisure Passenger (%)", 0.0, 20.0, 1.5, 0.1,
    help="Each leisure passenger who shows up has this chance of volunteering to give up their seat.",
)
vol_voucher = st.sidebar.number_input(
    "Voluntary Denied Boarding — Voucher ($)", 0, 5000, 800, 50,
    help="Voucher given to each volunteer. Airline keeps the original ticket revenue.",
)
invol_cost = st.sidebar.number_input(
    "Involuntary Denied Boarding — Total Cost ($)", 0, 20000, 3000, 100,
    help="$1,200 higher voucher + $1,800 estimated loss of goodwill. Airline keeps original ticket revenue.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
n_sims = st.sidebar.selectbox("Monte Carlo Iterations", [10_000, 50_000, 100_000], index=0)
random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Search Range")
max_tickets_low = st.sidebar.number_input(
    "Min Total Tickets to Sell", total_seats, total_seats + 100, total_seats, 1,
)
max_tickets_high = st.sidebar.number_input(
    "Max Total Tickets to Sell", total_seats, total_seats + 100, total_seats + 40, 1,
)
reservation_low = st.sidebar.number_input(
    "Min F1 Reservation Level", 0, 100, 10, 1,
    help="Minimum seats reserved for business fare passengers.",
)
reservation_high = st.sidebar.number_input(
    "Max F1 Reservation Level", 0, 100, 40, 1,
    help="Maximum seats reserved for business fare passengers.",
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Booking sequence**\n\n"
    "1. Leisure (F2) tickets sell first, up to the booking limit\n"
    "2. Business (F1) tickets sell in the final 2 weeks, up to total tickets\n"
    "3. On departure day, no-shows are resolved\n"
    "4. If oversold: leisure volunteers are sought first, then involuntary denial\n"
    "5. Business passengers never volunteer"
)

# ── Convert percentages ───────────────────────────────────────────────────────
f1_noshow = f1_noshow_pct / 100.0
f2_noshow = f2_noshow_pct / 100.0
volunteer_prob = volunteer_prob_pct / 100.0


# ── Simulation Engine ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_simulation(
    total_seats, f1_price, f2_price,
    f1_demand_mean, f1_demand_std, f2_demand_mean, f2_demand_std,
    f1_noshow, f2_noshow, volunteer_prob,
    vol_voucher, invol_cost,
    n_sims, random_seed,
    max_tickets_low, max_tickets_high,
    reservation_low, reservation_high,
):
    rng = np.random.default_rng(random_seed)
    records = []

    for total_tickets in range(max_tickets_low, max_tickets_high + 1):
        for f1_reserved in range(reservation_low, reservation_high + 1):
            booking_limit_f2 = total_tickets - f1_reserved  # max leisure tickets

            # --- Demand ---
            f2_demand = np.clip(
                np.round(rng.normal(f2_demand_mean, f2_demand_std, size=n_sims)).astype(int), 0, None
            )
            f1_demand = np.clip(
                np.round(rng.normal(f1_demand_mean, f1_demand_std, size=n_sims)).astype(int), 0, None
            )

            # --- Tickets sold ---
            f2_sold = np.minimum(f2_demand, booking_limit_f2)
            f1_sold = np.minimum(f1_demand, f1_reserved)

            # --- No-shows ---
            f2_show = rng.binomial(f2_sold, 1.0 - f2_noshow)
            f1_show = rng.binomial(f1_sold, 1.0 - f1_noshow)
            f1_noshow_count = f1_sold - f1_show
            total_show = f1_show + f2_show

            # --- Revenue ---
            # F2: airline keeps revenue even for no-shows
            f2_revenue = f2_price * f2_sold
            # F1: no-shows get full refund, so revenue = only those who show (or sold - noshow refund)
            f1_revenue = f1_price * (f1_sold - f1_noshow_count)  # = f1_price * f1_show

            revenue = f1_revenue + f2_revenue

            # --- Oversold situation ---
            excess = np.maximum(0, total_show - total_seats)

            # Volunteers: only leisure passengers who showed up can volunteer
            # Each has volunteer_prob chance
            potential_volunteers = rng.binomial(f2_show, volunteer_prob)
            actual_volunteers = np.minimum(potential_volunteers, excess)

            # Involuntary denied boardings: remaining excess after volunteers
            involuntary = np.maximum(0, excess - actual_volunteers)

            # --- Costs ---
            vol_cost_total = vol_voucher * actual_volunteers
            invol_cost_total = invol_cost * involuntary

            # --- Empty seats ---
            empty_seats = np.maximum(0, total_seats - total_show)

            # --- Profit ---
            profit = revenue - vol_cost_total - invol_cost_total

            records.append(dict(
                total_tickets=total_tickets,
                f1_reserved=f1_reserved,
                booking_limit_f2=booking_limit_f2,
                mean_profit=profit.mean(),
                std_profit=profit.std(),
                p05=np.percentile(profit, 5),
                p25=np.percentile(profit, 25),
                median=np.median(profit),
                p75=np.percentile(profit, 75),
                p95=np.percentile(profit, 95),
                mean_revenue=revenue.mean(),
                mean_f1_sold=f1_sold.mean(),
                mean_f2_sold=f2_sold.mean(),
                mean_f1_show=f1_show.mean(),
                mean_f2_show=f2_show.mean(),
                mean_total_show=total_show.mean(),
                mean_excess=excess.mean(),
                mean_volunteers=actual_volunteers.mean(),
                mean_involuntary=involuntary.mean(),
                prob_oversold=(excess > 0).mean(),
                prob_involuntary=(involuntary > 0).mean(),
                mean_empty=empty_seats.mean(),
                mean_vol_cost=vol_cost_total.mean(),
                mean_invol_cost=invol_cost_total.mean(),
                mean_f2_spilled=(f2_demand - f2_sold).mean(),
                mean_f1_spilled=(f1_demand - f1_sold).mean(),
            ))

    return pd.DataFrame(records)


with st.spinner("Simulating all combinations of total tickets and reservation levels…"):
    df = run_simulation(
        total_seats, f1_price, f2_price,
        f1_demand_mean, f1_demand_std, f2_demand_mean, f2_demand_std,
        f1_noshow, f2_noshow, volunteer_prob,
        vol_voucher, invol_cost,
        n_sims, random_seed,
        max_tickets_low, max_tickets_high,
        reservation_low, reservation_high,
    )

# ── Optimal ─────────────────────────────────────────────────────────────────────
opt_idx = df["mean_profit"].idxmax()
opt = df.loc[opt_idx]

# ── Single source-of-truth simulation at optimal ───────────────────────────────
@st.cache_data(show_spinner=False)
def get_optimal_dist(
    total_seats, total_tickets, f1_reserved,
    f1_price, f2_price, f1_noshow, f2_noshow,
    f1_demand_mean, f1_demand_std, f2_demand_mean, f2_demand_std,
    volunteer_prob, vol_voucher, invol_cost, n_sims, random_seed,
):
    rng = np.random.default_rng(random_seed)
    bl_f2 = total_tickets - f1_reserved
    f2_demand = np.clip(np.round(rng.normal(f2_demand_mean, f2_demand_std, n_sims)).astype(int), 0, None)
    f1_demand = np.clip(np.round(rng.normal(f1_demand_mean, f1_demand_std, n_sims)).astype(int), 0, None)
    f2_sold = np.minimum(f2_demand, bl_f2)
    f1_sold = np.minimum(f1_demand, f1_reserved)
    f2_show = rng.binomial(f2_sold, 1.0 - f2_noshow)
    f1_show = rng.binomial(f1_sold, 1.0 - f1_noshow)
    f1_noshow_ct = f1_sold - f1_show
    revenue = f2_price * f2_sold + f1_price * (f1_sold - f1_noshow_ct)
    total_show = f1_show + f2_show
    excess = np.maximum(0, total_show - total_seats)
    volunteers = np.minimum(rng.binomial(f2_show, volunteer_prob), excess)
    involuntary = np.maximum(0, excess - volunteers)
    profit = revenue - vol_voucher * volunteers - invol_cost * involuntary
    return profit, volunteers, involuntary, excess

dist, dist_vol, dist_invol, dist_excess = get_optimal_dist(
    total_seats, int(opt["total_tickets"]), int(opt["f1_reserved"]),
    f1_price, f2_price, f1_noshow, f2_noshow,
    f1_demand_mean, f1_demand_std, f2_demand_mean, f2_demand_std,
    volunteer_prob, vol_voucher, invol_cost, n_sims, random_seed,
)

# baseline: no overbooking (total_tickets = total_seats), min F1 reservation
baseline_row = df.loc[
    (df["total_tickets"] == total_seats) & (df["f1_reserved"] == reservation_low)
]
baseline_profit = baseline_row["mean_profit"].values[0] if not baseline_row.empty else df["mean_profit"].min()
gain = np.mean(dist) - baseline_profit

# ── KPI Cards ──────────────────────────────────────────────────────────────────
st.subheader("Optimal Strategy")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Tickets to Sell", f"{int(opt['total_tickets'])}")
k2.metric("F1 (Business) Reservation", f"{int(opt['f1_reserved'])} seats")
k3.metric("F2 (Leisure) Booking Limit", f"{int(opt['booking_limit_f2'])} tickets")
k4.metric("Expected Profit", f"${np.mean(dist):,.0f}", f"+${gain:,.0f} vs. baseline")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Avg. Volunteers", f"{np.mean(dist_vol):.2f}")
k6.metric("Avg. Involuntary Denied", f"{np.mean(dist_invol):.2f}")
k7.metric("Prob. of Oversold Flight", f"{(dist_excess > 0).mean()*100:.1f}%")
k8.metric("Prob. of Involuntary Denial", f"{(dist_invol > 0).mean()*100:.1f}%")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Profit Heatmap",
    "Profit Curves",
    "Overbooking & Volunteers",
    "Profit Distribution",
    "Full Results Table",
])

# --- Tab 1: Heatmap ---
with tab1:
    pivot = df.pivot_table(
        index="f1_reserved", columns="total_tickets", values="mean_profit",
    )
    fig_hm = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn",
        colorbar=dict(title="Mean Profit ($)"),
        hovertemplate=(
            "Total Tickets: %{x}<br>"
            "F1 Reserved: %{y}<br>"
            "Mean Profit: $%{z:,.0f}<extra></extra>"
        ),
    ))
    fig_hm.add_trace(go.Scatter(
        x=[int(opt["total_tickets"])], y=[int(opt["f1_reserved"])],
        mode="markers", marker=dict(size=14, color="red", symbol="star"),
        name=f"Optimal ({int(opt['total_tickets'])}, {int(opt['f1_reserved'])})",
    ))
    fig_hm.update_layout(
        title="Expected Profit by Total Tickets Sold & F1 Reservation Level",
        xaxis_title="Total Tickets to Sell",
        yaxis_title="Seats Reserved for Business (F1)",
        height=550,
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.info(
        f"The red star marks the optimal combination: sell **{int(opt['total_tickets'])}** "
        f"total tickets with **{int(opt['f1_reserved'])}** reserved for business passengers "
        f"(booking limit of **{int(opt['booking_limit_f2'])}** for leisure)."
    )

# --- Tab 2: Profit Curves ---
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Profit vs. Total Tickets (at optimal F1 reservation)")
        slice_res = df[df["f1_reserved"] == int(opt["f1_reserved"])].copy()
        fig_c1 = go.Figure()
        fig_c1.add_trace(go.Scatter(
            x=pd.concat([slice_res["total_tickets"], slice_res["total_tickets"][::-1]]),
            y=pd.concat([slice_res["p95"], slice_res["p05"][::-1]]),
            fill="toself", fillcolor="rgba(65,105,225,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="5th–95th %ile", hoverinfo="skip",
        ))
        fig_c1.add_trace(go.Scatter(
            x=slice_res["total_tickets"], y=slice_res["mean_profit"],
            mode="lines+markers", name="Mean Profit",
            line=dict(color="royalblue", width=2.5), marker=dict(size=4),
        ))
        fig_c1.add_vline(
            x=int(opt["total_tickets"]), line_dash="dash", line_color="red",
            annotation_text=f"Optimal: {int(opt['total_tickets'])}",
            annotation_font_color="red",
        )
        fig_c1.update_layout(
            xaxis_title="Total Tickets to Sell",
            yaxis_title="Expected Profit ($)",
            hovermode="x unified", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_c1, use_container_width=True)

    with col_b:
        st.markdown("#### Profit vs. F1 Reservation (at optimal total tickets)")
        slice_tix = df[df["total_tickets"] == int(opt["total_tickets"])].copy()
        fig_c2 = go.Figure()
        fig_c2.add_trace(go.Scatter(
            x=pd.concat([slice_tix["f1_reserved"], slice_tix["f1_reserved"][::-1]]),
            y=pd.concat([slice_tix["p95"], slice_tix["p05"][::-1]]),
            fill="toself", fillcolor="rgba(34,139,34,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="5th–95th %ile", hoverinfo="skip",
        ))
        fig_c2.add_trace(go.Scatter(
            x=slice_tix["f1_reserved"], y=slice_tix["mean_profit"],
            mode="lines+markers", name="Mean Profit",
            line=dict(color="forestgreen", width=2.5), marker=dict(size=4),
        ))
        fig_c2.add_vline(
            x=int(opt["f1_reserved"]), line_dash="dash", line_color="red",
            annotation_text=f"Optimal: {int(opt['f1_reserved'])}",
            annotation_font_color="red",
        )
        fig_c2.update_layout(
            xaxis_title="Seats Reserved for Business (F1)",
            yaxis_title="Expected Profit ($)",
            hovermode="x unified", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_c2, use_container_width=True)

# --- Tab 3: Overbooking & Volunteers ---
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Denied Boardings vs. Total Tickets")
        slice_r = df[df["f1_reserved"] == int(opt["f1_reserved"])].copy()
        fig_ob = go.Figure()
        fig_ob.add_trace(go.Bar(
            x=slice_r["total_tickets"], y=slice_r["mean_volunteers"],
            name="Avg. Volunteers (VDB)", marker_color="mediumseagreen", opacity=0.8,
        ))
        fig_ob.add_trace(go.Bar(
            x=slice_r["total_tickets"], y=slice_r["mean_involuntary"],
            name="Avg. Involuntary (IDB)", marker_color="salmon", opacity=0.8,
        ))
        fig_ob.add_vline(
            x=int(opt["total_tickets"]), line_dash="dash", line_color="red",
            annotation_text="Optimal", annotation_font_color="red",
        )
        fig_ob.update_layout(
            barmode="stack",
            xaxis_title="Total Tickets to Sell",
            yaxis_title="Avg. Denied Boardings",
            hovermode="x unified", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ob, use_container_width=True)

    with col2:
        st.markdown("#### Denied Boarding Costs vs. Total Tickets")
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=slice_r["total_tickets"], y=slice_r["mean_vol_cost"],
            mode="lines+markers", name=f"Volunteer Cost (${vol_voucher}/ea)",
            line=dict(color="mediumseagreen", width=2.5), marker=dict(size=4),
        ))
        fig_cost.add_trace(go.Scatter(
            x=slice_r["total_tickets"], y=slice_r["mean_invol_cost"],
            mode="lines+markers", name=f"Involuntary Cost (${invol_cost}/ea)",
            line=dict(color="salmon", width=2.5), marker=dict(size=4),
        ))
        fig_cost.add_vline(
            x=int(opt["total_tickets"]), line_dash="dash", line_color="red",
            annotation_text="Optimal", annotation_font_color="red",
        )
        fig_cost.update_layout(
            xaxis_title="Total Tickets to Sell",
            yaxis_title="Avg. Cost ($)",
            hovermode="x unified", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown(
        f"**Voluntary Denied Boarding (VDB):** Leisure passengers who volunteer to take "
        f"the next flight in exchange for an **${vol_voucher}** voucher. Each showing leisure "
        f"passenger has a **{volunteer_prob_pct:.1f}%** chance of volunteering.\n\n"
        f"**Involuntary Denied Boarding (IDB):** When there aren't enough volunteers, "
        f"passengers are involuntarily bumped at a cost of **${invol_cost}** each "
        f"($1,200 voucher + $1,800 estimated goodwill loss)."
    )

# --- Tab 4: Profit Distribution at Optimal ---
with tab4:

    col_h, col_s = st.columns([2, 1])
    with col_h:
        fig_dist = px.histogram(
            dist, nbins=60, color_discrete_sequence=["steelblue"],
            labels={"value": "Profit ($)", "count": "Frequency"},
            title=f"Profit Distribution at Optimal Strategy ({n_sims:,} flights)",
        )
        fig_dist.add_vline(
            x=float(np.mean(dist)), line_dash="dash", line_color="red",
            annotation_text=f"Mean = ${np.mean(dist):,.0f}",
        )
        fig_dist.add_vline(
            x=float(np.percentile(dist, 5)), line_dash="dot", line_color="orange",
            annotation_text=f"5th %ile = ${np.percentile(dist, 5):,.0f}",
        )
        fig_dist.update_layout(
            showlegend=False, xaxis_title="Profit ($)", yaxis_title="Frequency", height=450,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_s:
        st.markdown("#### Summary at Optimal")
        summary = {
            "Total Tickets": f"{int(opt['total_tickets'])}",
            "F1 Reserved": f"{int(opt['f1_reserved'])}",
            "F2 Booking Limit": f"{int(opt['booking_limit_f2'])}",
            "Mean Profit": f"${np.mean(dist):,.0f}",
            "Std Deviation": f"${np.std(dist):,.0f}",
            "5th Percentile": f"${np.percentile(dist, 5):,.0f}",
            "Median": f"${np.median(dist):,.0f}",
            "95th Percentile": f"${np.percentile(dist, 95):,.0f}",
            "Avg. Volunteers": f"{np.mean(dist_vol):.2f}",
            "Avg. Involuntary": f"{np.mean(dist_invol):.2f}",
            "Prob. Oversold": f"{(dist_excess > 0).mean()*100:.1f}%",
        }
        for label, val in summary.items():
            st.markdown(f"**{label}:** {val}")

# --- Tab 5: Full Results Table ---
with tab5:
    display = df[[
        "total_tickets", "f1_reserved", "booking_limit_f2",
        "mean_profit", "std_profit", "p05", "p95",
        "mean_f1_sold", "mean_f2_sold",
        "mean_volunteers", "mean_involuntary",
        "prob_oversold", "prob_involuntary",
        "mean_empty",
    ]].copy()
    display.columns = [
        "Total Tickets", "F1 Reserved", "F2 Booking Limit",
        "Mean Profit", "Std Dev", "5th %ile", "95th %ile",
        "Avg F1 Sold", "Avg F2 Sold",
        "Avg Volunteers", "Avg Involuntary",
        "Prob Oversold", "Prob Involuntary",
        "Avg Empty Seats",
    ]
    for col in ["Mean Profit", "Std Dev", "5th %ile", "95th %ile"]:
        display[col] = display[col].map("${:,.0f}".format)
    for col in ["Avg F1 Sold", "Avg F2 Sold", "Avg Volunteers", "Avg Involuntary", "Avg Empty Seats"]:
        display[col] = display[col].map("{:.2f}".format)
    for col in ["Prob Oversold", "Prob Involuntary"]:
        display[col] = display[col].map("{:.1%}".format)

    st.dataframe(display, use_container_width=True, hide_index=True, height=500)

st.markdown("---")

# ── How It Works ───────────────────────────────────────────────────────────────
with st.expander("How This Model Works"):
    st.markdown(f"""
**Two Decision Variables:**
1. **Total tickets to sell** — can exceed {total_seats} seats (overbooking)
2. **F1 reservation level** — seats reserved for business passengers

The **F2 booking limit** = total tickets − F1 reservation.

**Booking Sequence:**
- Leisure (F2) passengers book first (up to the booking limit)
- Business (F1) passengers book in the final 2 weeks (up to the reservation level)

**On Departure Day:**
- Each F2 passenger has a **{f2_noshow_pct:.1f}%** no-show rate (airline keeps revenue)
- Each F1 passenger has a **{f1_noshow_pct:.1f}%** no-show rate (airline refunds the ticket)

**If Oversold (more show-ups than {total_seats} seats):**
1. Ask leisure passengers to volunteer — each has a **{volunteer_prob_pct:.1f}%** chance
   → Volunteers get an **${vol_voucher}** voucher (airline keeps original ticket revenue)
2. If not enough volunteers → involuntary denied boarding
   → Costs **${invol_cost}** per passenger ($1,200 voucher + $1,800 goodwill loss)
3. Business passengers never volunteer

**Demand Distributions:**
- F1 (Business): N({f1_demand_mean}, {f1_demand_std}) — rounded to nearest integer
- F2 (Leisure): N({f2_demand_mean}, {f2_demand_std}) — rounded to nearest integer
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Airline Revenue Management Simulator · "
    f"Monte Carlo with {n_sims:,} iterations · Seed: {random_seed}"
)

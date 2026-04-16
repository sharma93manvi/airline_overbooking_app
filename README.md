# ✈️ Airline Overbooking Simulation Platform

A Streamlit app that uses Monte Carlo simulation and analytical methods to find the optimal number of tickets an airline should sell above seat capacity to maximize expected profit.

## Features

- Monte Carlo simulation with configurable iterations (10k–100k)
- Closed-form analytical expected profit calculation using the binomial distribution
- Interactive charts: expected profit curves with confidence bands, bumping risk, and profit distribution histogram
- Sidebar controls for seats, pricing, voucher cost, no-show probability, and simulation settings

## Quick Start

```bash
pip install streamlit numpy pandas plotly scipy
streamlit run airline_overbooking_app.py
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| Number of Seats | Aircraft seat capacity |
| Seat Price | Revenue per ticket sold |
| Voucher Cost | Compensation per bumped passenger |
| No-Show Probability | Chance a ticketed passenger doesn't show up |

## How It Works

Each passenger shows up independently with probability `1 - no_show_prob`. The simulation sells `N + overbook` tickets, simulates arrivals via the binomial distribution, and computes net profit after voucher costs for bumped passengers. The optimal overbooking level is the one that maximizes expected profit.

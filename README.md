# 📊 Automated Traffic Analytics using Agentic AI

## Overview

This project is an **automated analytics system** that reads structured monthly traffic data from Excel and generates **professional, point-in-time analytical commentary** using an agent-based AI architecture.

The system is designed to **compare measured data points only** (e.g., month-over-month, same-month year-over-year, and year-to-date), while explicitly avoiding long-term trend inference to prevent misleading conclusions when data is limited.

The final insights and metrics are written back into the same Excel file in a **report-ready format**.

---

## Key Objectives

- Automate calculation of:
  - Adjacent-month (MoM) change
  - Same-month year-over-year (YoY) change
  - Year-to-date (YTD) comparison
- Generate **controlled, factual analytical narratives**
- Avoid hallucinated trends or unsupported claims
- Ensure narratives remain **numerically defensible**
- Maintain Excel as both **input and output interface**

---

## Why Point-in-Time Analysis?

This system intentionally **does not perform trend analysis**.

A trend implies sustained movement over time, which requires:
- complete historical data
- sufficient time span
- statistical validation

Instead, this system performs **point-to-point comparisons**, which are:
- accurate with partial data
- suitable for monthly reporting
- analytically responsible

Every generated summary explicitly clarifies that it reflects a **two-point comparison**, not a sustained trend.

---

## Architecture (Agent-Based Design)

The system is built using **LangGraph**, with each agent having a single responsibility:

### 1️⃣ Metrics Agent
- Computes:
  - Month-over-month (LM)
  - Same-month year-over-year (YoY)
  - Year-to-date (YTD)
- Counts:
  - Number of positive vs negative LM movements
- Calculates:
  - Average absolute LM (volatility measure)

### 2️⃣ Reasoning Agent
- Interprets numeric comparisons only
- Compares relative magnitude and direction
- Prohibited from describing trends, causes, or momentum

### 3️⃣ Planning Agent
- Determines narrative emphasis (LM vs YoY)
- Controls language strength based on numeric magnitude
- Detects directional mismatches (LM vs YoY)

### 4️⃣ Narrative Agent
- Produces structured, professional commentary
- Uses strict language rules
- Ensures:
  - No trend inference
  - No causal assumptions
  - No hallucinated insights

---

## Key Metrics Explained

### Adjacent-Month Change (LM)
Compares the latest month with the immediately previous month.

### Same-Month Year-over-Year (YoY)
Compares the same calendar month across two years (e.g., August 2025 vs August 2024).

### Year-to-Date Change (YTD)
Compares cumulative values from January up to the latest measured month against the same period last year.

### Average Absolute LM
- Measures **volatility**, not direction
- Uses absolute values to avoid implying improvement or decline
- Helps quantify how large month-to-month movements typically are

### Positive vs Negative LM Count
- Shows distribution of monthly movement
- Avoids claiming overall direction
- Adds context without trend inference

---

## Input & Output

### Input
- Excel file (`input.xlsx`)
- Multiple traffic tables inside a single sheet
- Monthly data for multiple segments

### Output
- Calculated metrics written back to Excel
- AI-generated analytical summaries written into merged cells
- Fully report-ready format

---

## Technologies Used

- **Python**
- **Pandas** – data handling
- **OpenPyXL** – Excel read/write and formatting
- **LangChain** – LLM integration
- **LangGraph** – agent orchestration
- **OpenAI GPT model** – controlled narrative generation

---

## How This Project Is Different

- No black-box analytics
- No fabricated insights
- No trend hallucinations
- Clear separation between computation and interpretation
- Designed with **analytical correctness over narrative flashiness**

---

## Disclaimer

All generated insights reflect **measured data points only**.  
They do not represent forecasts, trends, or long-term behavioral conclusions.

---

## Author

**Aryan Tyagi**  
B.Tech CSE (AI & ML)  
Agentic Analytics | Data Interpretation | Automation

---

> *“The goal of this system is not to sound insightful, but to be analytically correct.”*

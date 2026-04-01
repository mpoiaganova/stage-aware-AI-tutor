import pandas as pd
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("chatbot_sample.csv")

# -----------------------------
# Config
# -----------------------------
TZ = "Europe/Zurich"
FONT = "Arial"

BASE_FONT = 18
TITLE_FONT = 20
AXIS_TITLE_FONT = 20
TICK_FONT = 16
LEGEND_FONT = 18
BUBBLE_LABEL_FONT = 16

# -----------------------------
# Load + prep
# -----------------------------
df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True, errors="coerce")
df = df.dropna(subset=["time_stamp"])

df["time_local"] = df["time_stamp"].dt.tz_convert(TZ)
df["date_local"] = df["time_local"].dt.date

df = df[df["source"].astype(str).str.lower().eq("user")].copy()

course_start = df["date_local"].min()
df["course_day"] = (
    (pd.to_datetime(df["date_local"]) - pd.to_datetime(course_start)).dt.days + 1
).astype(int)

# Activity mapping
df["Doing_what"] = pd.to_numeric(df["Doing_what"], errors="coerce")
df.loc[df["Doing_what"] == 33, "Doing_what"] = 3

label_map = {
    0: "Other learning activity",
    1: "Self-study",
    2: "Quizzes",
    3: "Analytics project",
}
df = df[df["Doing_what"].isin(label_map.keys())].copy()
df["Activity"] = df["Doing_what"].map(label_map)

# -----------------------------
# Average per person (include zeros)
# -----------------------------
users = np.sort(df["user_Id"].dropna().unique())
last_active_day = int(df["course_day"].max())
all_days = np.arange(1, last_active_day + 1)
activities = list(label_map.values())

counts = (
    df.groupby(["course_day", "user_Id", "Activity"], as_index=False)
      .size()
      .rename(columns={"size": "n_messages"})
)

grid = pd.MultiIndex.from_product(
    [all_days, users, activities],
    names=["course_day", "user_Id", "Activity"]
).to_frame(index=False)

full = grid.merge(counts, on=["course_day", "user_Id", "Activity"], how="left")
full["n_messages"] = full["n_messages"].fillna(0)

avg = (
    full.groupby(["course_day", "Activity"], as_index=False)["n_messages"]
        .mean()
        .rename(columns={"n_messages": "avg_messages_per_user"})
)

avg["label"] = np.where(
    avg["avg_messages_per_user"] > 10,
    avg["avg_messages_per_user"].round(1).astype(str),
    ""
)

avg_nz = avg[avg["avg_messages_per_user"] > 0].copy()

# -----------------------------
# Plot settings
# -----------------------------
activity_order = ["Self-study", "Quizzes", "Other learning activity", "Analytics project"]
y_order_bottom_to_top = ["Analytics project", "Other learning activity", "Quizzes", "Self-study"]

color_map = {
    "Self-study": "#FF9913",
    "Quizzes": "#3EB489",
    "Other learning activity": "#875692",
    "Analytics project": "#66B3FF",
}

max_val = max(1e-9, avg_nz["avg_messages_per_user"].max())
avg_nz["size_px"] = np.sqrt(avg_nz["avg_messages_per_user"] / max_val) * 60
avg_nz["size_px"] = avg_nz["size_px"].clip(lower=8)

# -----------------------------
# Build figure
# -----------------------------
fig = go.Figure()

for act in activity_order:
    d = avg_nz[avg_nz["Activity"] == act]
    fig.add_trace(go.Scatter(
        x=d["course_day"],
        y=d["Activity"],
        mode="markers+text",
        name=act,
        text=d["label"],
        textposition="middle center",
        textfont=dict(size=BUBBLE_LABEL_FONT, family=FONT, color="black"),
        marker=dict(
            size=d["size_px"],
            color=color_map[act],
            opacity=0.85,
            line=dict(width=1.2, color="rgba(0,0,0,0.20)")
        ),
        customdata=d["avg_messages_per_user"],
        hovertemplate=(
            "Day %{x}<br>"
            "Activity: %{y}<br>"
            "Avg messages per user: %{customdata:.2f}<extra></extra>"
        )
    ))

# -----------------------------
# Layout
# -----------------------------
fig.update_layout(
    template="plotly_white",
    width=950,
    height=440,
    font=dict(family=FONT, size=BASE_FONT),
    title=dict(
        text="Learning Activities Across Course Days",
        x=0.5,
        xanchor="center",
        font=dict(size=TITLE_FONT)
    ),
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=1.08,
        yanchor="bottom",
        itemsizing="constant",
        font=dict(size=LEGEND_FONT),
        title=""
    ),
    margin=dict(l=135, r=40, t=120, b=80),
)

fig.update_xaxes(
    title="Course day",
    tickmode="array",
    tickvals=list(all_days),     # <- explicit ticks only
    ticktext=[str(d) for d in all_days],
    range=[0.5, last_active_day + 0.5],
    tickfont=dict(size=TICK_FONT),
    title_font=dict(size=AXIS_TITLE_FONT),
    gridcolor="rgba(0,0,0,0.07)",
    zeroline=False
)

fig.update_yaxes(
    categoryorder="array",
    categoryarray=y_order_bottom_to_top,
    tickfont=dict(size=TICK_FONT),
    gridcolor="rgba(0,0,0,0.07)",
    title=""
)

fig.show()

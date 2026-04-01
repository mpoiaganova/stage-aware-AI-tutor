import pandas as pd
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("chatbot_sample.csv")

# ----------------------------
# CONFIG
# ----------------------------
user_col = "participant_id"
thread_col = "Thread"       
ts_col = "time_stamp"
bot_col = "bot_type"     
source_col = "source"       

TZ = "Europe/Zurich"
FONT = "Arial"

BASE_FONT = 28
TITLE_FONT = 30
AXIS_TITLE_FONT = 28
TICK_FONT = 18
LEGEND_FONT = 22

COLOR_T = "#3EB489"
COLOR_C = "#C64B8C"

FIG_W, FIG_H = 1400, 720
MARGIN = dict(l=95, r=60, t=170, b=110)

TEXT_THRESHOLD = 1

# Grid styling: thin grey lines on WHITE background
GRID_LINE_COLOR = "rgba(0,0,0,0.10)"
GRID_LINE_WIDTH = 1

# ----------------------------
# LOAD / CLEAN
# ----------------------------
df = df.copy()

df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
df = df.dropna(subset=[ts_col, user_col, bot_col]).copy()

# keep only USER messages so cells show # messages sent by user
df = df[df[source_col].astype(str).str.lower().eq("user")].copy()

# local day
df[ts_col] = df[ts_col].dt.tz_convert(TZ)
df["day_local"] = df[ts_col].dt.floor("D")

# course day index (1..N)
course_start = df["day_local"].min()
df["course_day"] = (df["day_local"] - course_start).dt.days + 1

# group
df["group"] = np.where(df[bot_col] == "(C)", "C", "T")

# ----------------------------
# MESSAGE COUNTS (messages per user-day-group)
# ----------------------------
ud = (
    df.groupby([user_col, "course_day", "group"], as_index=False)
      .size()
      .rename(columns={"size": "n_messages"})
)

# ----------------------------
# USER ORDER: sort by total activity
# ----------------------------
user_totals = (
    ud.groupby(user_col)["n_messages"]
      .sum()
      .sort_values(ascending=False)
)
users_sorted = user_totals.index.tolist()

user_to_label = {u: f"U{i+1}" for i, u in enumerate(users_sorted)}
ud["user_label"] = ud[user_col].map(user_to_label)
user_labels = [user_to_label[u] for u in users_sorted]

# course day range
last_day = int(ud["course_day"].max())
all_days = np.arange(1, last_day + 1)
x_labels = list(all_days)

# ----------------------------
# MATRICES
# ----------------------------
cnt_mat = (
    ud.pivot_table(index="user_label", columns="course_day", values="n_messages", fill_value=0)
      .reindex(index=user_labels, columns=all_days, fill_value=0)
)

grp_mat = (
    ud.pivot_table(index="user_label", columns="course_day", values="group", aggfunc="first")
      .reindex(index=user_labels, columns=all_days)
)

# ----------------------------
# INTENSITY 
# ----------------------------
max_cnt = cnt_mat.to_numpy().max()
intensity = np.sqrt(cnt_mat / max_cnt) if max_cnt > 0 else cnt_mat * 0.0

text_mat = cnt_mat.astype(int).astype(str).where(cnt_mat >= TEXT_THRESHOLD, "")

# ----------------------------
# PLOT (heatmap layers + text overlay)
# ----------------------------
fig = go.Figure()

# Treatment heatmap
fig.add_trace(go.Heatmap(
    z=intensity.where(grp_mat == "T").to_numpy(),
    x=x_labels,
    y=user_labels,
    colorscale=[[0, "white"], [1, COLOR_T]],
    zmin=0, zmax=1,
    showscale=False,
    hovertemplate="User=%{y}<br>Course day=%{x}<br>Bot=Treatment (T)<br>Messages=%{customdata}<extra></extra>",
    customdata=cnt_mat.to_numpy().astype(int),
    showlegend=False
))

# Control heatmap
fig.add_trace(go.Heatmap(
    z=intensity.where(grp_mat == "C").to_numpy(),
    x=x_labels,
    y=user_labels,
    colorscale=[[0, "white"], [1, COLOR_C]],
    zmin=0, zmax=1,
    showscale=False,
    hovertemplate="User=%{y}<br>Course day=%{x}<br>Bot=Control (C)<br>Messages=%{customdata}<extra></extra>",
    customdata=cnt_mat.to_numpy().astype(int),
    showlegend=False
))

# Text overlay (counts)
fig.add_trace(go.Heatmap(
    z=np.zeros_like(intensity.to_numpy()),
    x=x_labels,
    y=user_labels,
    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
    showscale=False,
    text=text_mat.to_numpy(),
    texttemplate="%{text}",
    textfont=dict(size=TICK_FONT, family=FONT, color="rgba(0,0,0,0.9)"),
    hoverinfo="skip",
    showlegend=False
))

# Legend markers (dummy traces)
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=14, color=COLOR_T),
    name="Scaffolding",
    hoverinfo="skip",
    showlegend=True
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=14, color=COLOR_C),
    name="Baseline",
    hoverinfo="skip",
    showlegend=True
))

# ----------------------------
# CELL GRID 
# ----------------------------
n_users = len(user_labels)
shapes = []

# vertical grid lines at boundaries: 0.5, 1.5, ..., last_day + 0.5
for xb in np.arange(0.5, last_day + 1.5, 1):
    shapes.append(dict(
        type="line",
        xref="x", yref="y",
        x0=xb, x1=xb,
        y0=-0.5, y1=n_users - 0.5,
        line=dict(color=GRID_LINE_COLOR, width=GRID_LINE_WIDTH),
        layer="below"
    ))

# horizontal grid lines at boundaries: -0.5, 0.5, ..., n_users - 0.5
for yb in np.arange(-0.5, n_users + 0.5, 1):
    shapes.append(dict(
        type="line",
        xref="x", yref="y",
        x0=0.5, x1=last_day + 0.5,
        y0=yb, y1=yb,
        line=dict(color=GRID_LINE_COLOR, width=GRID_LINE_WIDTH),
        layer="below"
    ))

# ----------------------------
# LAYOUT
# ----------------------------
fig.update_layout(
    template="plotly_white",
    width=FIG_W,
    height=FIG_H,
    margin=MARGIN,
    font=dict(family=FONT, size=BASE_FONT),
    title=dict(
        text="Individual Usage as Number of Messages Sent",
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
        title="AI tutor type"
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    shapes=shapes
)

fig.update_xaxes(
    title="Course day",
    tickmode="array",
    tickvals=list(all_days),
    ticktext=[str(d) for d in all_days],
    range=[0.5, last_day + 0.5],
    tickfont=dict(size=TICK_FONT),
    title_font=dict(size=AXIS_TITLE_FONT),
    showgrid=False,
    zeroline=False
)

fig.update_yaxes(
    title="User",
    autorange="reversed",
    categoryorder="array",
    categoryarray=user_labels,
    tickfont=dict(size=TICK_FONT),
    title_font=dict(size=AXIS_TITLE_FONT),
    showgrid=False,
    zeroline=False
)

fig.show()

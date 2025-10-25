import os
import random
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

import psycopg2
import pandas as pd
import plotly.express as px
import streamlit as st

from substrateinterface import SubstrateInterface

REFERENCE_VALIDATOR = "5GZ2KuT2TtLbYTtsMcgAtazo6KQ4bc57ykZgyQv9oit3y7iq"
DEFAULT_NETUID = 5
DEFAULT_ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"
SS58_FORMAT = 42
U16_MAX = 65535


def fmt_ts(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "unknown"
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    local_dt = dt.astimezone()
    return (
        f"{dt.strftime('%Y-%m-%d %H:%M:%S %Z')} / "
        f"{local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )


def human_delta(ms_then: int | None, ms_now: int | None) -> str:
    if ms_then is None or ms_now is None:
        return "unknown"
    delta_s = max(0, (ms_now - ms_then) / 1000.0)
    mins, secs = divmod(int(delta_s), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def connect_substrate(endpoint: str = DEFAULT_ENDPOINT):
    if SubstrateInterface is None:
        raise RuntimeError("substrate-interface not installed (chain unavailable)")
    return SubstrateInterface(
        url=endpoint,
        ss58_format=SS58_FORMAT,
        use_remote_preset=True,
    )


def read_weights_by_uid(substrate: SubstrateInterface, netuid: int, uid: int):
    q = substrate.query("SubtensorModule", "Weights", [netuid, uid])
    vals = q.value or []
    return [(int(dest), int(w)) for dest, w in vals]


def get_last_update_block(substrate: SubstrateInterface, netuid: int, uid: int) -> int | None:
    q = substrate.query("SubtensorModule", "LastUpdate", [netuid], block_hash=None)
    arr = q.value
    if not arr or uid >= len(arr):
        return None
    try:
        return int(arr[uid])
    except Exception:
        try:
            return int(arr[uid].value)
        except Exception:
            return None


def get_block_timestamp(substrate: SubstrateInterface, block_number: int) -> int | None:
    if block_number is None:
        return None
    try:
        block_hash = substrate.get_block_hash(block_number)
        if block_hash is None:
            return None
        ts_q = substrate.query("Timestamp", "Now", block_hash=block_hash)
        ts = ts_q.value
        return int(ts) if ts is not None else None
    except Exception:
        return None


def get_current_block_and_timestamp(substrate: SubstrateInterface) -> tuple[int | None, int | None]:
    header = substrate.get_block_header()
    try:
        current_block = int(header["header"]["number"])
    except Exception:
        current_block = None
    ts_q = substrate.query("Timestamp", "Now")
    current_ts = int(ts_q.value) if ts_q and ts_q.value is not None else None
    return current_block, current_ts


def get_validator_uid(substrate: SubstrateInterface, netuid: int, hotkey: str) -> int | None:
    try:
        q = substrate.query("SubtensorModule", "Uids", [netuid, hotkey])
        if q and q.value is not None:
            return int(q.value)
        return None
    except Exception:
        return None


def get_hotkey_for_uid(substrate: SubstrateInterface, netuid: int, uid: int) -> Optional[str]:
    try:
        q = substrate.query("SubtensorModule", "Keys", [netuid, uid])
        if q and q.value is not None:
            return str(q.value)
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def resolve_hotkeys_for_uids(uids: List[int], netuid: int = DEFAULT_NETUID, endpoint: str = DEFAULT_ENDPOINT) -> Dict[int, str]:
    out: Dict[int, str] = {}
    try:
        substrate = connect_substrate(endpoint)
    except Exception:
        return out
    for uid in uids:
        hk = get_hotkey_for_uid(substrate, netuid, uid)
        if hk:
            out[uid] = hk
    return out


def summarize_last_set_time(
    last_block: int | None,
    last_ts_ms: int | None,
    current_block: int | None,
    current_ts_ms: int | None,
    avg_block_seconds: float = 12.0,
):
    if last_block is not None and current_block is not None:
        blocks_since = current_block - last_block
        if blocks_since < 0:
            blocks_since = 0
    else:
        blocks_since = None
    if last_ts_ms is not None and current_ts_ms is not None:
        last_ts_human = fmt_ts(last_ts_ms)
        time_since = human_delta(last_ts_ms, current_ts_ms)
        return {
            "last_ts_human": last_ts_human,
            "time_since": time_since,
            "blocks_since": blocks_since,
        }
    if (
        last_ts_ms is None
        and current_ts_ms is not None
        and blocks_since is not None
        and blocks_since >= 0
    ):
        est_delta_seconds = blocks_since * avg_block_seconds
        est_then_ms = current_ts_ms - int(est_delta_seconds * 1000)
        last_ts_human = fmt_ts(est_then_ms)
        time_since = human_delta(est_then_ms, current_ts_ms)
        return {
            "last_ts_human": last_ts_human + " (est)",
            "time_since": time_since + " (est)",
            "blocks_since": blocks_since,
        }
    return {
        "last_ts_human": "unknown",
        "time_since": "unknown",
        "blocks_since": blocks_since,
    }


def fetch_validator_weights_info(uid: int, netuid: int, endpoint: str, top_n: int = 25):
    try:
        substrate = connect_substrate(endpoint)
        current_block, current_ts_ms = get_current_block_and_timestamp(substrate)
        weights = read_weights_by_uid(substrate, netuid, uid)
        last_block = get_last_update_block(substrate, netuid, uid)
        last_ts_ms = get_block_timestamp(substrate, last_block)
        timing = summarize_last_set_time(
            last_block=last_block,
            last_ts_ms=last_ts_ms,
            current_block=current_block,
            current_ts_ms=current_ts_ms,
            avg_block_seconds=12.0,
        )
        weights_sorted = sorted(weights, key=lambda x: x[1], reverse=True)
        total = sum(w for _, w in weights_sorted) or 1
        top_rows = []
        for dest_uid, w in weights_sorted[:top_n]:
            share = w / total
            top_rows.append(
                {
                    "dest_uid": dest_uid,
                    "weight_ticks": w,
                    "share": share,
                    "percent": share * 100.0,
                }
            )
        weights_df = pd.DataFrame(top_rows)
        info = {
            "current_block": current_block,
            "current_ts_human": fmt_ts(current_ts_ms),
            "last_block": last_block,
            "last_ts_human": timing["last_ts_human"],
            "time_since": timing["time_since"],
            "blocks_since": timing["blocks_since"],
            "weights_df": weights_df,
        }
        return info, None
    except Exception as e:
        return None, str(e)


def summarize_last_set_weights_for_hotkey(hotkey: str, endpoint: str = DEFAULT_ENDPOINT):
    try:
        substrate = connect_substrate(endpoint)
    except Exception:
        return {"last_ts_human": "unknown", "time_since": "unknown"}
    uid = get_validator_uid(substrate, DEFAULT_NETUID, hotkey)
    if uid is None:
        return {"last_ts_human": "unknown", "time_since": "unknown"}
    current_block, current_ts_ms = get_current_block_and_timestamp(substrate)
    last_block = get_last_update_block(substrate, DEFAULT_NETUID, uid)
    last_ts_ms = get_block_timestamp(substrate, last_block)
    timing = summarize_last_set_time(
        last_block=last_block,
        last_ts_ms=last_ts_ms,
        current_block=current_block,
        current_ts_ms=current_ts_ms,
        avg_block_seconds=12.0,
    )
    return {
        "last_ts_human": timing["last_ts_human"],
        "time_since": timing["time_since"],
    }


def _mock_now():
    return datetime.utcnow()


def _mock_overview_df():
    rows = []
    base_block = 100000
    uids = [101, 202, 303]
    for uid in uids:
        for i in range(80):
            ts = _mock_now() - timedelta(minutes=i)
            block = base_block + i
            response_time = random.uniform(0.05, 0.8)
            exact_flag = random.random() < 0.4 + (0.2 if uid == 202 else 0.0)
            partial_correctness = random.uniform(0.2, 0.95)
            grid_similarity = random.uniform(0.3, 0.99)
            efficiency_score = random.uniform(0.4, 1.0)
            rows.append(
                {
                    "ts": ts,
                    "block": block,
                    "uid": uid,
                    "problem_id": f"prob-{uid}-{i}",
                    "success": True,
                    "response_time": response_time,
                    "exact_match": exact_flag,
                    "partial_correctness": partial_correctness,
                    "grid_similarity": grid_similarity,
                    "efficiency_score": efficiency_score,
                    "accuracy": 1 if exact_flag else 0,
                }
            )
    return pd.DataFrame(rows)


def _mock_validator_versions_df():
    latest_version = "v1.3.7"
    data = [
        {
            "wallet_hotkey": REFERENCE_VALIDATOR,
            "version": latest_version,
            "cycle_count": 1234,
            "ts": _mock_now() - timedelta(seconds=15),
        },
        {
            "wallet_hotkey": "5OtherHotkey11111111111111111111111111111111",
            "version": latest_version,
            "cycle_count": 1180,
            "ts": _mock_now() - timedelta(minutes=5),
        },
        {
            "wallet_hotkey": "5OutdatedKey2222222222222222222222222222222",
            "version": "v1.2.9",
            "cycle_count": 900,
            "ts": _mock_now() - timedelta(minutes=25),
        },
    ]
    df = pd.DataFrame(data)
    return df, latest_version


def _mock_miners_list(df_overview):
    return sorted(df_overview["uid"].unique().tolist())


def _mock_miner_stats_df(df_overview, uid):
    d = df_overview[df_overview["uid"] == uid]
    if d.empty:
        return pd.DataFrame(
            [],
            columns=[
                "uid",
                "total_queries",
                "accuracy",
                "avg_latency_s",
                "avg_partial",
                "avg_grid_sim",
                "avg_efficiency",
            ],
        )
    total_queries = len(d)
    accuracy = d["accuracy"].mean()
    avg_latency = d["response_time"].mean()
    avg_partial = d["partial_correctness"].mean()
    avg_grid = d["grid_similarity"].mean()
    avg_eff = d["efficiency_score"].mean()
    row = {
        "uid": uid,
        "total_queries": total_queries,
        "accuracy": accuracy,
        "avg_latency_s": avg_latency,
        "avg_partial": avg_partial,
        "avg_grid_sim": avg_grid,
        "avg_efficiency": avg_eff,
    }
    return pd.DataFrame([row])


def _mock_recent_for_miner(df_overview, uid, limit=50):
    d = df_overview[df_overview["uid"] == uid].copy()
    d = d.sort_values("ts", ascending=False).head(limit)
    d["accuracy"] = d["accuracy"].astype(int)
    return d[
        [
            "ts",
            "block",
            "problem_id",
            "accuracy",
            "response_time",
            "partial_correctness",
            "grid_similarity",
            "efficiency_score",
        ]
    ]


@st.cache_resource(show_spinner=False)
def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None
    try:
        parsed = urlparse(db_url)
        conn = psycopg2.connect(
            dbname=parsed.path.lstrip("/"),
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port or 5432,
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"[dashboard] Failed DB connect, mock mode: {e}")
        return None


def sql_df(query, params=None):
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("DB not available")
    with conn.cursor() as cur:
        cur.execute(query, params or [])
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def get_miners_list(full_df_overview=None):
    try:
        df = sql_df("SELECT DISTINCT uid FROM miner_metrics ORDER BY uid;")
        return df["uid"].tolist()
    except Exception:
        if full_df_overview is not None and not full_df_overview.empty:
            return _mock_miners_list(full_df_overview)
        return []


def get_overview_data(days_window: int, selected_miners):
    try:
        base_q = """
            SELECT
                ts,
                block,
                uid,
                problem_id,
                success,
                response_time,
                exact_match,
                partial_correctness,
                grid_similarity,
                efficiency_score
            FROM miner_metrics
            WHERE ts >= NOW() - (%s || ' days')::interval
        """
        params = [str(days_window)]
        if selected_miners:
            base_q += " AND uid = ANY(%s)"
            params.append(selected_miners)
        base_q += " ORDER BY ts DESC LIMIT 5000"
        df = sql_df(base_q, params)
        if df.empty:
            return df
        df["accuracy"] = df["exact_match"].astype(int)
        return df
    except Exception:
        df = _mock_overview_df()
        cutoff = datetime.utcnow() - timedelta(days=days_window)
        df = df[df["ts"] >= cutoff]
        if selected_miners:
            df = df[df["uid"].isin(selected_miners)]
        return df


def get_miner_stats(uid, fallback_df=None):
    try:
        q = """
            SELECT
                uid,
                COUNT(*) AS total_queries,
                AVG(CASE WHEN exact_match THEN 1 ELSE 0 END) AS accuracy,
                AVG(response_time) AS avg_latency_s,
                AVG(partial_correctness) AS avg_partial,
                AVG(grid_similarity) AS avg_grid_sim,
                AVG(efficiency_score) AS avg_efficiency
            FROM miner_metrics
            WHERE uid = %s
        GROUP BY uid
        """
        df = sql_df(q, [uid])
        return df
    except Exception:
        if fallback_df is None or fallback_df.empty:
            fallback_df = _mock_overview_df()
        return _mock_miner_stats_df(fallback_df, uid)


def get_miner_recent_samples(uid, limit=50, fallback_df=None):
    try:
        q = """
            SELECT
                ts,
                block,
                problem_id,
                exact_match,
                response_time,
                partial_correctness,
                grid_similarity,
                efficiency_score
            FROM miner_metrics
            WHERE uid = %s
            ORDER BY ts DESC
            LIMIT %s
        """
        df = sql_df(q, [uid, limit])
        if not df.empty:
            df["accuracy"] = df["exact_match"].astype(int)
        return df
    except Exception:
        if fallback_df is None or fallback_df.empty:
            fallback_df = _mock_overview_df()
        return _mock_recent_for_miner(fallback_df, uid, limit=limit)


def get_validator_versions():
    try:
        q = """
            SELECT DISTINCT ON (wallet_hotkey)
                wallet_hotkey,
                version,
                cycle_count,
                ts
            FROM validator_heartbeat
            ORDER BY wallet_hotkey, ts DESC
        """
        df = sql_df(q)
        q2 = """
            SELECT version
            FROM validator_heartbeat
            WHERE wallet_hotkey = %s
            ORDER BY ts DESC
            LIMIT 1
        """
        ref_df = sql_df(q2, [REFERENCE_VALIDATOR])
        latest_version = ref_df.iloc[0]["version"] if not ref_df.empty else None
        return df, latest_version
    except Exception:
        df, latest_version = _mock_validator_versions_df()
        return df, latest_version


def parse_semver(v: str | None):
    if not v:
        return None
    v = v.lstrip("vV")
    parts = v.split(".")
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            out.append(None)
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])


def version_status_label_and_color(v: str | None, latest: str | None):
    if v is None or latest is None:
        return ("unknown", "⚪")
    if v == latest:
        return ("up to date", "🟢")
    pv = parse_semver(v)
    pl = parse_semver(latest)
    if pv is None or pl is None:
        return ("behind", "🟠")
    v_maj, v_min, v_patch = pv
    l_maj, l_min, l_patch = pl
    if v_maj == l_maj and v_min == l_min:
        if (v_patch or 0) < (l_patch or 0):
            return ("slightly behind", "🟠")
        else:
            return ("unknown", "⚪")
    else:
        return ("far behind", "🔴")


def page_overview():
    st.title("🌐 Overview: Miner Performance")

    top_row = st.columns([1, 5, 1])
    with top_row[0]:
        if st.button("Refresh data"):
            st.rerun()

    control_row = st.columns([1, 2])
    with control_row[0]:
        days_window = st.slider(
            "Time window (last X days)",
            min_value=1,
            max_value=7,
            value=1,
            step=1,
            help="Show only queries newer than now() - X days",
        )

    mock_df_full = _mock_overview_df()
    with control_row[1]:
        miner_list = get_miners_list(mock_df_full)
        selected_miners = st.multiselect(
            "Filter by miner UID(s)",
            options=miner_list,
            default=[],
        )

    df = get_overview_data(days_window, selected_miners)

    if df.empty:
        st.warning("No data found for this selection yet.")
        return

    agg = (
        df.groupby("uid", as_index=False)
        .agg(
            avg_latency_s=("response_time", "mean"),
            accuracy=("accuracy", "mean"),
            queries=("uid", "size"),
        )
    )

    uid_list = agg["uid"].tolist()
    hotkey_map = resolve_hotkeys_for_uids(uid_list)
    agg["hotkey"] = agg["uid"].map(lambda u: hotkey_map.get(u, ""))

    search_term = st.text_input(
        "Search by UID or hotkey (live filter)",
        value="",
        help="Type part of a UID or hotkey address to filter the chart/table.",
    )

    if search_term:
        s = str(search_term).lower()
        mask = (
            agg["uid"].astype(str).str.contains(s, case=False)
            | agg["hotkey"].astype(str).str.lower().str.contains(s, case=False)
        )
        agg_filtered = agg[mask].copy()
    else:
        agg_filtered = agg.copy()

    st.subheader("Accuracy vs Latency (Per Miner)")
    st.caption("Each point is one miner UID. Accuracy = mean(exact_match). Latency = mean(response_time in s).")

    if agg_filtered.empty:
        st.info("No miners match this filter.")
    else:
        fig_scatter = px.scatter(
            agg_filtered,
            x="avg_latency_s",
            y="accuracy",
            size="queries",
            color=agg_filtered["uid"].astype(str),
            hover_data=[
                "uid",
                "hotkey",
                "queries",
                "avg_latency_s",
                "accuracy",
            ],
            labels={
                "avg_latency_s": "Avg Latency (s)",
                "accuracy": "Exact Match Rate",
                "queries": "# Queries",
                "color": "Miner UID",
            },
            title="Miner Accuracy vs Latency",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    st.subheader("Miner summary table")
    nice_table = agg_filtered.copy()
    nice_table["accuracy_pct"] = (nice_table["accuracy"] * 100.0).round(2)
    nice_table["avg_latency_ms"] = (nice_table["avg_latency_s"] * 1000.0).round(2)
    nice_table = nice_table[
        [
            "uid",
            "hotkey",
            "queries",
            "accuracy_pct",
            "avg_latency_s",
        ]
    ].rename(
        columns={
            "uid": "UID",
            "hotkey": "Hotkey",
            "queries": "Total queries",
            "accuracy_pct": "Accuracy (%)",
            "avg_latency_s": "Avg latency (s)",
        }
    )
    st.dataframe(nice_table, use_container_width=True)


def page_miner_detail():
    st.title("🔎 Miner Detail")

    top_row = st.columns([1, 5, 1])
    with top_row[0]:
        if st.button("Refresh miner data"):
            st.rerun()

    mock_df_full = _mock_overview_df()
    miner_list = get_miners_list(mock_df_full)

    if not miner_list:
        st.warning("No miner data yet.")
        return

    uid = st.selectbox("Select miner UID", options=miner_list)

    stats_df = get_miner_stats(uid, fallback_df=mock_df_full)
    st.subheader("Summary stats")

    if stats_df.empty:
        st.info("No stats yet for this miner.")
    else:
        row = stats_df.iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total queries", int(row["total_queries"]))
        with col2:
            acc_pct = (
                float(row["accuracy"]) * 100 if row["accuracy"] is not None else 0.0
            )
            st.metric("Exact match rate", f"{acc_pct:.2f}%")
        with col3:
            st.metric("Avg latency (s)", f"{row['avg_latency_s']:.3f}")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Avg partial correctness", f"{row['avg_partial']:.3f}")
        with col5:
            st.metric("Avg grid similarity", f"{row['avg_grid_sim']:.3f}")
        with col6:
            st.metric("Avg efficiency", f"{row['avg_efficiency']:.3f}")

    st.markdown("---")
    st.subheader("Recent queries from this miner")

    recent_df = get_miner_recent_samples(uid, fallback_df=mock_df_full)
    if recent_df.empty:
        st.info("No recent samples.")
    else:
        st.caption(
            "We NEVER reveal the correct target output. "
            "Only miner predictions & timing."
        )

        st.dataframe(
            recent_df[
                [
                    "ts",
                    "block",
                    "problem_id",
                    "accuracy",
                    "response_time",
                    "partial_correctness",
                    "grid_similarity",
                    "efficiency_score",
                ]
            ],
            use_container_width=True,
        )

        st.subheader("Accuracy vs Latency (this miner)")
        miner_fig = px.scatter(
            recent_df,
            x="response_time",
            y=recent_df["accuracy"].astype(int),
            hover_data=["problem_id", "block", "ts"],
            labels={
                "response_time": "Latency (s)",
                "y": "Exact Match",
            },
            title=f"Miner {uid}: Accuracy vs Latency",
        )
        st.plotly_chart(miner_fig, use_container_width=True)


def page_validators():
    st.title("🛡 Validator Status")

    top_row = st.columns([1, 5, 1])
    with top_row[0]:
        if st.button("Refresh validators"):
            st.rerun()

    df, latest_version = get_validator_versions()

    if df.empty:
        st.warning("No validator heartbeat data yet.")
        return

    status_labels = []
    status_dots = []
    last_heartbeat_strs = []
    for _, row in df.iterrows():
        lbl, dot = version_status_label_and_color(
            row.get("version"), latest_version
        )
        status_labels.append(lbl)
        status_dots.append(dot)

        ts_val = row.get("ts")
        if isinstance(ts_val, datetime):
            last_heartbeat_strs.append(ts_val.strftime("%Y-%m-%d %H:%M:%S UTC"))
        else:
            last_heartbeat_strs.append(str(ts_val))

    df["status_label"] = status_labels
    df["status_dot"] = status_dots
    df["last_heartbeat"] = last_heartbeat_strs

    last_set_times = []
    time_since_set = []
    for _, row in df.iterrows():
        hotkey = row["wallet_hotkey"]
        info = summarize_last_set_weights_for_hotkey(hotkey, DEFAULT_ENDPOINT)
        last_set_times.append(info["last_ts_human"])
        time_since_set.append(info["time_since"])
    df["last_set_weights_at"] = last_set_times
    df["since_last_set"] = time_since_set

    st.subheader("Latest validator state")
    st.caption(
        f"Latest expected version comes from {REFERENCE_VALIDATOR[:6]}..."
        f"{REFERENCE_VALIDATOR[-6:]}"
    )
    st.metric("Latest expected version", latest_version or "unknown")

    table_df = pd.DataFrame(
        {
            "validator": df["wallet_hotkey"],
            "version": df["version"],
            "status": df["status_dot"] + " " + df["status_label"],
            "last_heartbeat": df["last_heartbeat"],
            "last_set_weights_at": df["last_set_weights_at"],
            "since_last_set": df["since_last_set"],
        }
    )

    st.dataframe(
        table_df,
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("Validator Inspector")
    st.caption(
        "Inspect a validator's current weights vector. netuid is fixed to 5."
    )

    colA, colB = st.columns([1, 2])
    with colA:
        validator_uid = st.number_input(
            "Validator UID (chain index)",
            min_value=0,
            max_value=10000,
            value=251,
            step=1,
        )
    with colB:
        endpoint = st.text_input(
            "Subtensor endpoint",
            value=DEFAULT_ENDPOINT,
            help="WebSocket endpoint for the subnet chain",
        )

    if st.button("Fetch on-chain weights info"):
        info_box = st.empty()
        weights_box = st.empty()

        info, err = fetch_validator_weights_info(
            uid=int(validator_uid),
            netuid=DEFAULT_NETUID,
            endpoint=endpoint,
            top_n=25,
        )

        if err:
            info_box.error(f"Error fetching chain data: {err}")
        else:
            info_box.markdown(
                f"""
**Validator UID {validator_uid} (netuid {DEFAULT_NETUID})**

- Last set_weights block: `{info['last_block']}`
- Last set_weights time: `{info['last_ts_human']}`
- Time since last set_weights: `{info['time_since']}`
- Blocks since last set_weights: `{info['blocks_since']}`  
- Current chain block: `{info['current_block']}`  
- Current chain time: `{info['current_ts_human']}`
                """
            )

            if info["weights_df"].empty:
                weights_box.info("No weights found (validator may not have set any).")
            else:
                weights_box.dataframe(
                    info["weights_df"],
                    use_container_width=True,
                )


st.set_page_config(
    page_title="Hone Telemetry Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Miner Detail", "Validators"],
    index=0,
)

if page == "Overview":
    page_overview()
elif page == "Miner Detail":
    page_miner_detail()
elif page == "Validators":
    page_validators()

import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer
import faiss
from pymongo import MongoClient
import os
import re
import datetime
import pandas as pd
import pymssql
import plotly.express as px

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB = os.environ.get("MONGO_DB", "vehicle_attendance")
COLL = os.environ.get("MONGO_COLLECTION", "chatbot_docs")

# SQL DB (for vehicle report tab)
DB_SERVER = os.environ.get("DB_SERVER", "")
DB_NAME = os.environ.get("DB_NAME", "")
DB_USER = os.environ.get("DB_USER", "")
DB_PASS = os.environ.get("DB_PASS", "")

# load once (RAG resources)
index = faiss.read_index(INDEX_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
mongo = MongoClient(MONGO_URI)[DB][COLL]

STRICT_REFUSAL_THRESHOLD = float(os.environ.get("RAG_STRICT_THRESHOLD", "0.35"))  # cosine/IP score


def retrieve(q, k=3):
    """Retrieve top-k chunks with scores from FAISS+Mongo.

    Returns a list of dicts: {"text": str, "score": float} sorted by relevance.
    """
    q = (q or "").strip()
    if not q:
        return []

    emb = embedder.encode([q], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    D, I = index.search(emb, k)

    ids = [int(x) for x in I[0] if int(x) >= 0]
    if not ids:
        return []

    docs = list(mongo.find({"faiss_idx": {"$in": ids}}))
    docs_map = {int(d["faiss_idx"]): d for d in docs}

    results = []
    for idx, score in zip(I[0], D[0]):
        idx = int(idx)
        if idx < 0:
            continue
        d = docs_map.get(idx)
        if not d:
            continue
        txt = (d.get("text") or "").strip()
        if not txt:
            continue
        results.append({
            "text": txt,
            "score": float(score),
        })
    return results


IDK_MESSAGE = "I'm sorry, but I don't know the answer based on the company data I have."


# --- Vehicle report helpers (SQL) ---

def normalize_vehicle(v: str) -> str:
    if v is None:
        return ""
    v = v.upper()
    v = re.sub(r"[\s\-]", "", v)
    return v


SQL = r"""
WITH vqr AS (
    SELECT vqrId, VehicalNumber
    FROM Vehical_QR_Master WITH (NOLOCK)
    WHERE VehicalNumber = %s
),
gc_union AS (
    SELECT gcDate, vehicleNumber, houseId, gcType, 0 as isNotScan
    FROM GarbageCollectionDetails WITH (NOLOCK)
    WHERE CAST(gcDate AS DATE) BETWEEN %s AND %s

    UNION ALL

    SELECT gcDate, vehicleNumber, houseId, gcType, 1 as isNotScan
    FROM GarbageCollection_NotScan WITH (NOLOCK)
    WHERE CAST(gcDate AS DATE) BETWEEN %s AND %s
),
filtered_gc AS (
    SELECT G.*, hm.ZoneId, wd.PanelId, G.vehicleNumber
    FROM gc_union G
    LEFT JOIN HouseMaster hm ON hm.houseId = G.houseId
    LEFT JOIN WardNumber wd ON hm.WardNo = wd.Id
    WHERE (%s = 0 OR %s IS NULL OR hm.ZoneId = %s)
      AND (%s = 0 OR %s IS NULL OR wd.PanelId = %s)
      AND G.vehicleNumber = %s
),
attendance AS (
    SELECT CAST(DA.daDate AS DATE) AS dt,
           DA.daID,
           DA.VQRId,
           DA.startTime,
           DA.endTime
    FROM Daily_Attendance DA WITH (NOLOCK)
    WHERE CAST(DA.daDate AS DATE) BETWEEN %s AND %s
      AND DA.VQRId = (SELECT vqrId FROM vqr)
)
SELECT
    A.dt AS Date,
    V.VehicalNumber AS VehicleNumber,
    MIN(CASE WHEN G.gcType = 1 THEN CAST(G.gcDate AS TIME) END) AS FirstHouseScan,
    MAX(CASE WHEN G.gcType = 1 THEN CAST(G.gcDate AS TIME) END) AS LastHouseScan,
    SUM(CASE WHEN G.gcType = 1 THEN 1 ELSE 0 END) AS TotalHouseCount,
    MAX(CASE WHEN G.gcType = 3 THEN CAST(G.gcDate AS TIME) END) AS LastDumpScan,
    SUM(CASE WHEN G.gcType = 3 THEN 1 ELSE 0 END) AS TotalDumpTrip,
    MIN(A.startTime) AS DutyOnTime,
    MAX(A.endTime) AS DutyOffTime
FROM attendance A
LEFT JOIN vqr V ON V.vqrId = A.VQRId
LEFT JOIN filtered_gc G ON CAST(G.gcDate AS DATE) = A.dt AND G.vehicleNumber = V.VehicalNumber
GROUP BY A.dt, V.VehicalNumber
ORDER BY A.dt ASC;
"""


def get_db_conn():
    if not (DB_SERVER and DB_NAME and DB_USER and DB_PASS):
        raise RuntimeError("DB_SERVER, DB_NAME, DB_USER, DB_PASS env vars must be set for vehicle report.")
    return pymssql.connect(
        server=f"{DB_SERVER},1433",
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8",
    )


def get_vehicle_daily_stats(vehicle_number, from_date, to_date, zone_id=0, panel_id=0):
    v = normalize_vehicle(vehicle_number)
    params = [
        v,  # Vehical_QR_Master.VehicalNumber = %s
        from_date, to_date,  # gc_union first part
        from_date, to_date,  # gc_union second part
        zone_id, zone_id, zone_id,  # filtered_gc zone placeholders (three)
        panel_id, panel_id, panel_id,  # filtered_gc panel placeholders (three)
        v,  # filtered_gc vehicleNumber match
        from_date, to_date,  # attendance date filter
    ]
    conn = get_db_conn()
    try:
        df = pd.read_sql(SQL, conn, params=params)
    finally:
        conn.close()

    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        numeric_cols = ["TotalHouseCount", "TotalDumpTrip"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def row_to_rag_fact(row):
    parts = []
    parts.append(f"On {row['Date']} vehicle {row['VehicleNumber']}")
    if row.get("TotalHouseCount", None) is not None:
        parts.append(f"scanned {int(row['TotalHouseCount'])} houses")
    if row.get("TotalDumpTrip", None) is not None:
        parts.append(f"and did {int(row['TotalDumpTrip'])} dump trips")
    if row.get("FirstHouseScan", None):
        parts.append(f"first scan at {row['FirstHouseScan']}")
    if row.get("LastHouseScan", None):
        parts.append(f"last scan at {row['LastHouseScan']}")
    if row.get("DutyOnTime", None) and row.get("DutyOffTime", None):
        parts.append(f"duty {row['DutyOnTime']} - {row['DutyOffTime']}")
    return ". ".join(parts) + "."


# streamlit run app.py --server.port 7860

def rag(q):
    """RAG answer with strict refusals and a human, but grounded, tone."""
    ctx = retrieve(q)

    # If nothing relevant is retrieved, refuse to answer.
    if not ctx:
        return IDK_MESSAGE

    # Use the best score (FAISS returns results sorted by score).
    best_score = ctx[0]["score"]
    if best_score < STRICT_REFUSAL_THRESHOLD:
        return IDK_MESSAGE

    ctx_text = "\n\n".join(c["text"] for c in ctx)

    prompt = (
        "You are a friendly but precise company assistant. "
        "You have access only to the CONTEXT below, which comes from internal databases and documents. "
        "You MUST obey these rules:\n"
        "1) Use ONLY the information in the context to answer. Do NOT use outside knowledge.\n"
        "2) If the answer is not clearly supported by the context, or you are unsure, "
        "you MUST reply exactly: 'I don't know the answer based on the company data I have.'\n"
        "3) When you do answer, be short, human, and easy to read. Prefer 1–3 concise sentences or a VERY short bullet list.\n\n"
        f"CONTEXT:\n{ctx_text}\n\n"
        f"QUESTION: {q}\n"
        "Now give your answer following the rules above."
    )

    r = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"]

st.set_page_config(page_title="Trashbot", layout="wide")

st.markdown("""
<h1 style="font-size:42px; font-weight:900; margin-bottom:0px;">Trashbot</h1>
<p style="font-size:14px; color:#888; margin-top:-5px;">by Rishikesh</p>
""", unsafe_allow_html=True)

chat_tab, report_tab = st.tabs(["Chatbot", "Vehicle Report"])

with chat_tab:
    st.subheader("Company Data Chatbot")
    st.markdown(
        """
        <p style="color:#555; font-size:14px;">
        Hi, I'm your Trashbot assistant. I answer questions based <strong>only</strong> on your company data
        (indexed from your databases and documents). If something is not in the data, I'll tell you that
        instead of guessing.
        </p>
        <p style="color:#777; font-size:13px;">
        <em>Examples:</em> "How many houses did vehicle MH08-AP-1894 cover yesterday?",<br/>
        "What was the duty time for vehicle MH08-AP-1894 on 2024-06-05?"
        </p>
        """,
        unsafe_allow_html=True,
    )
    query = st.text_input("Ask a question about the company data:")
    if query:
        with st.spinner("Thinking based on your company data..."):
            ans = rag(query)
        st.write(ans)

with report_tab:
    st.subheader("Vehicle Duty / Scan Report")
    st.write("Query raw data directly from the database (read-only).")

    col1, col2 = st.columns(2)
    with col1:
        vehicle_input = st.text_input("Vehicle Number (free text)", value="")
        zone_id = st.number_input("ZoneId (0 = all)", value=0, step=1)
        panel_id = st.number_input("PanelId (0 = all)", value=0, step=1)
    with col2:
        from_date = st.date_input(
            "From date",
            value=(datetime.date.today() - datetime.timedelta(days=7)),
        )
        to_date = st.date_input("To date", value=datetime.date.today())

    st.caption(
        "DB credentials are taken from environment variables DB_SERVER/DB_NAME/DB_USER/DB_PASS."
    )

    run = st.button("Fetch report")

    if run:
        if not vehicle_input.strip():
            st.error("Enter vehicle number (e.g. MH08-AP-1894).")
        else:
            with st.spinner("Querying database..."):
                try:
                    df = get_vehicle_daily_stats(
                        vehicle_input,
                        from_date.strftime("%Y-%m-%d"),
                        to_date.strftime("%Y-%m-%d"),
                        int(zone_id),
                        int(panel_id),
                    )
                except Exception as e:
                    st.exception(e)
                    df = pd.DataFrame()

            if df.empty:
                st.warning("No records found for that vehicle / date range.")
            else:
                st.subheader("Results")
                st.dataframe(
                    df[
                        [
                            "Date",
                            "VehicleNumber",
                            "DutyOnTime",
                            "FirstHouseScan",
                            "LastHouseScan",
                            "TotalHouseCount",
                            "LastDumpScan",
                            "TotalDumpTrip",
                            "DutyOffTime",
                        ]
                    ].sort_values("Date"),
                    use_container_width=True,
                )

                if "TotalHouseCount" in df.columns:
                    fig = px.bar(
                        df,
                        x="Date",
                        y="TotalHouseCount",
                        title=f"Daily House Count for {normalize_vehicle(vehicle_input)}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                rag_texts = [row_to_rag_fact(r) for _, r in df.iterrows()]
                rag_blob = "\n".join(rag_texts)
                st.download_button(
                    "Download facts for RAG (txt)",
                    data=rag_blob,
                    file_name=f"{normalize_vehicle(vehicle_input)}_facts.txt",
                    mime="text/plain",
                )

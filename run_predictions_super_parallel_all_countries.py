# run_predictions_async.py
import os, json, re, asyncio, math, time
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type
from openai import APIError, RateLimitError, APITimeoutError


# ---- Globaler Async Rate-Limiter für TPM & RPM 
class RateLimiter:
    def __init__(self, tpm_limit: int, rpm_limit: int, tokens_per_req: int):
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
        self.tokens_per_req = tokens_per_req

        self._bucket = float(tpm_limit)        # Token-Bucket (Capacity = TPM)
        self._rate = tpm_limit / 60.0          # Tokens, die pro Sekunde nachfließen
        self._last = time.monotonic()

        self._rpm_count = 0
        self._rpm_window_start = time.monotonic()

        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            # Bucket refillen
            self._bucket = min(self.tpm_limit, self._bucket + self._rate * (now - self._last))
            self._last = now

            need = self.tokens_per_req

            def rpm_blocked():
                return self._rpm_count >= self.rpm_limit and (time.monotonic() - self._rpm_window_start) < 60.0

            # warten, bis genug Tokens im Bucket & RPM-Fenster frei
            while self._bucket < need or rpm_blocked():
                wait_tokens = (need - self._bucket) / self._rate if self._bucket < need else 0.0
                wait_rpm = (60.0 - (time.monotonic() - self._rpm_window_start)) if rpm_blocked() else 0.0
                await asyncio.sleep(max(0.01, wait_tokens, wait_rpm))

                now = time.monotonic()
                self._bucket = min(self.tpm_limit, self._bucket + self._rate * (now - self._last))
                self._last = now

            # Budget reservieren
            self._bucket -= need

            # RPM-Fenster updaten
            now = time.monotonic()
            if (now - self._rpm_window_start) >= 60.0:
                self._rpm_window_start = now
                self._rpm_count = 0
            self._rpm_count += 1
# ----------------------------
# Konfiguration
# ----------------------------
DATA_PATH = Path("./data/ess11_full_filtered.csv")
META_PATH = Path("./data/ess11_full_variables.csv")
OUT_WIDE_CSV = Path("./data/ess11_full_llm_predictions_wide.csv")

MODEL = "gpt-5-nano"   
MAX_PERSONS: Optional[int] = None  # z.B. 200 für Teillauf
CONCURRENCY = 64        # maximale gleichzeitige API-Calls (global)
CHUNK_PERSONS = 4       # wie viele Personen parallel verarbeiten (4–16)
BATCH_WRITE_SIZE = 250  # nach so vielen Ergebnissen flushen
REQUEST_TIMEOUT = 30     # Sekunden pro Request

# Optionales Token-Throttling 
TOKENS_PER_REQ_EST = 1650  
TPM_BUDGET = 190000          # z.B. 200_000; None = aus
RPM_LIMIT = 500

limiter = RateLimiter(TPM_BUDGET, RPM_LIMIT, TOKENS_PER_REQ_EST)
# ----------------------------

# .env laden & Client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY fehlt. Bitte in .env setzen.")
client = AsyncOpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT)

def load_inputs(data_path: Path, meta_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)
    meta = pd.read_csv(meta_path)

    if "unique_id" not in df.columns:
        raise ValueError("Column 'unique_id' missing in data CSV.")
    for c in ["var_name", "question_text"]:
        if c not in meta.columns:
            raise ValueError(f"Column '{c}' missing in meta CSV.")

    for c in ["answer_translation", "prediction_values", "options_ranges_list"]:
        if c not in meta.columns:
            meta[c] = ""

    for c in ["question_text", "answer_translation", "prediction_values", "options_ranges_list"]:
        meta[c] = meta[c].fillna("").astype(str)

    return df, meta

def build_system_prompt(meta_row: pd.Series) -> str:
    q_text = meta_row.get("question_text", "").strip()
    options = meta_row.get("options_ranges_list", "").strip()
    trans = meta_row.get("answer_translation", "").strip()

    parts = [
        "You are a model that predicts user responses to survey questions.",
        "",
        "Use the user's previous answers to select the most appropriate value for the target question.",
        "Only use the provided options for your answer. Do not include explanations, text, or any other formatting.",
        "Your response must be a single integer from the provided options or ranges.",
        "",
        "Note: The following special codes may appear in user responses:",
        "77 = Don't know",
        "88 = No answer",
        "99 = Refused",
        "These are context only and must never be predicted.",
        "",
        f"Target question: {q_text}",
        f"Options or ranges: {options}",
    ]
    if trans:
        parts.append("The following definitions explain what the numeric options represent for this question:")
        parts.append(trans)
    return "\n".join(parts)

def render_user_line(q_text: str, trans: str, value) -> str:
    val_str = ""
    if pd.notna(value) and str(value).strip() != "":
        try:
            value = int(float(value))
        except Exception:
            pass
        val_str = str(value)
    return f"{q_text} ({trans}): {val_str}" if trans else f"{q_text}: {val_str}"

def build_user_prompt(person_row: pd.Series, meta: pd.DataFrame, target_var: str) -> str:
    lines = []
    
    # Land des Users hinzufügen
    if "cntry" in person_row.index:
        country = person_row["cntry"]
        lines.append(f"Survey Country of user: {country}")
        lines.append("")
    
    lines.append("User responses:")
    for _, m in meta.iterrows():
        var = m["var_name"]
        if var == target_var:
            continue
        if var in person_row.index:
            q_text = m.get("question_text", var)
            trans = str(m.get("answer_translation", "") or "")
            v = person_row[var]
            lines.append(f"- {render_user_line(q_text, trans, v)}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)



# ---------- Async LLM Call mit Retries ----------
sema = asyncio.Semaphore(CONCURRENCY)

@retry(
    wait=wait_exponential_jitter(initial=0.5, max=8.0),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError))
)
async def call_llm_async(system_prompt: str, user_prompt: str) -> str:
    #token_throttle()
    await limiter.acquire()  # ← exakte Drossel vor jedem Call
    async with sema:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

async def run_all_async(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    meta_idx = meta.set_index("var_name")
    vars_list: List[str] = list(meta["var_name"])

    # Output vorbereiten / laden
    if OUT_WIDE_CSV.exists():
        out = pd.read_csv(OUT_WIDE_CSV, dtype=str)
    else:
        out = pd.DataFrame({"unique_id": df["unique_id"].astype(str)})
    for v in vars_list:
        col = f"pred_{v}"
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")

    results_buffer: List[Dict[str, str]] = []

    async def process_one(uid: str, person_row: pd.Series, var: str):
        mrow = meta_idx.loc[var]
        system_prompt = build_system_prompt(mrow)
        user_prompt = build_user_prompt(person_row, meta, var)
        raw = await call_llm_async(system_prompt, user_prompt)
        results_buffer.append({"unique_id": uid, "pred_col": f"pred_{var}", "value": raw})

    async def process_person(uid: str, person_row: pd.Series):
        # ganze Person überspringen, wenn pred_stflife schon da
        stflife_col = "pred_stflife"
        if stflife_col in out.columns:
            existing = out.loc[out["unique_id"] == uid, stflife_col]
            if not existing.empty and str(existing.iloc[0]).strip() not in ["", "nan", "NaN"]:
                print(f"⏭️  Überspringe {uid}: {stflife_col} bereits gefüllt.")
                return

        tasks = []
        for var in vars_list:
            if var in df.columns:
                tasks.append(process_one(uid, person_row, var))
        if tasks:
            await asyncio.gather(*tasks)

    async def flush_results():
        nonlocal results_buffer, out
        if not results_buffer:
            return
        # in DataFrame schreiben (synchroner Teil)
        for item in results_buffer:
            uid = item["unique_id"]
            col = item["pred_col"]
            val = item["value"]
            mask = out["unique_id"].astype(str) == str(uid)
            if not mask.any():
                new_row = {"unique_id": str(uid)}
                out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
                mask = out["unique_id"].astype(str) == str(uid)
            out.loc[mask, col] = str(val)
        out.to_csv(OUT_WIDE_CSV, index=False)
        results_buffer = []

    #mehrere Personen parallel (Chunking) 
    batch_tasks: List[asyncio.Task] = []
    for i, row in enumerate(df.itertuples(index=False)):
        if MAX_PERSONS is not None and i >= MAX_PERSONS:
            break
        person_row = pd.Series(row._asdict(), index=df.columns)
        uid = str(person_row.get("unique_id"))

        batch_tasks.append(asyncio.create_task(process_person(uid, person_row)))

        # Wenn genug Personen in der Luft: warten + flushen
        if len(batch_tasks) >= CHUNK_PERSONS:
            await asyncio.gather(*batch_tasks)
            await flush_results()
            print(f"💾 Flush nach Batch bis Person #{i+1}")
            batch_tasks = []

        # zusätzlicher Flush, falls viele Einzel-Ergebnisse gepuffert sind
        if len(results_buffer) >= BATCH_WRITE_SIZE:
            await flush_results()

    # Restliche Personen abwarten + final flush
    if batch_tasks:
        await asyncio.gather(*batch_tasks)
    await flush_results()
    print(f"✅ Gespeichert: {OUT_WIDE_CSV}")

def main():
    df, meta = load_inputs(DATA_PATH, META_PATH)
    asyncio.run(run_all_async(df, meta))

if __name__ == "__main__":
    main()

"""In-notebook manual rating widget (ipywidgets). Saves to CSV after every click, so it resumes."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

DIMS = ("relevance", "groundedness", "fluency")
RUBRIC = {
    "relevance": "5 = products & answer fit every stated need · 3 = partly · 1 = unrelated",
    "groundedness": "5 = every claim (price, colour, brand, features, IDs) is in the product cards · 1 = invented",
    "fluency": "5 = clear, natural, well organised · 3 = understandable but clumsy · 1 = broken",
}


def make_rater(sheet_path: str | Path, dims=DIMS):
    import ipywidgets as w
    from IPython.display import display

    sheet_path = Path(sheet_path)
    df = pd.read_csv(sheet_path)
    for d in dims:
        if d not in df.columns:
            df[d] = pd.NA
    if "notes" not in df.columns:
        df["notes"] = ""
    df["notes"] = df["notes"].fillna("").astype(str)

    def todo_idx():
        return [i for i in df.index if df.loc[i, list(dims)].isna().any()]

    state = {"i": (todo_idx() or [0])[0]}
    view = w.HTML()
    progress = w.HTML()
    toggles = {d: w.ToggleButtons(options=[1, 2, 3, 4, 5], value=None, description=d[:12],
                                  tooltips=[RUBRIC[d]] * 5, style={"button_width": "42px"}) for d in dims}
    notes = w.Text(placeholder="notes (optional)", layout=w.Layout(width="70%"))
    b_prev, b_save, b_skip = w.Button(description="◀ Prev"), w.Button(description="Save & next ▶",
                                                                        button_style="success"), w.Button(description="Skip")

    def render():
        i = state["i"]
        r = df.loc[i]
        prods = html.escape(str(r.get("products_text", ""))).replace("\n", "<br>")
        ans = html.escape(str(r.get("answer", ""))).replace("\n", "<br>")
        view.value = (f"<div style='font-family:sans-serif;max-width:900px'>"
                      f"<b>{html.escape(str(r.get('case_id', i)))}</b> &nbsp;<i>{html.escape(str(r.get('source', '')))}</i>"
                      f"<h4 style='margin:6px 0'>Q: {html.escape(str(r['query']))}</h4>"
                      f"<div style='background:#f6f6f6;padding:6px;font-size:12px'>{prods}</div>"
                      f"<div style='border-left:3px solid #4a7;padding:6px;margin-top:6px'>{ans}</div></div>")
        for d in dims:
            v = r[d]
            toggles[d].value = int(v) if pd.notna(v) else None
        notes.value = str(r.get("notes", "") or "")
        done = len(df) - len(todo_idx())
        progress.value = f"<b>{done}/{len(df)} rated</b> · row {i + 1} · rubric: " + " | ".join(
            f"<i>{d}</i>: {RUBRIC[d]}" for d in dims)

    def save(_=None):
        i = state["i"]
        for d in dims:
            if toggles[d].value is not None:
                df.loc[i, d] = toggles[d].value
        df.loc[i, "notes"] = notes.value
        df.to_csv(sheet_path, index=False)

    def go(step):
        state["i"] = (state["i"] + step) % len(df)
        render()

    def on_save(_):
        save()
        rest = todo_idx()
        if rest:
            later = [j for j in rest if j > state["i"]]
            state["i"] = (later or rest)[0]
            render()
        else:
            render()
            progress.value = "<b style='color:green'>All rows rated ✔ — saved to " + html.escape(str(sheet_path)) + "</b>"

    b_save.on_click(on_save)
    b_skip.on_click(lambda _: go(1))
    b_prev.on_click(lambda _: go(-1))
    render()
    display(w.VBox([progress, view, *toggles.values(), notes, w.HBox([b_prev, b_skip, b_save])]))
    return df

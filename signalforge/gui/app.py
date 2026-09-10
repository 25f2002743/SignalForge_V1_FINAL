from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from signalforge.ingestion.loader import load_signal
from signalforge.preprocessing.pipeline import prepare
from signalforge.observatory.engine import observe
from signalforge.parameters.estimate import extract_parameters
from signalforge.modulation.advisor import advise
from signalforge.hypotheses.generator import generate
from signalforge.hypotheses.scorer import rank_hypotheses
from signalforge.hypotheses.executor import execute_hypothesis
from signalforge.demod.engine import demodulate
from signalforge.bitstream.metrics import bit_statistics
from signalforge.deinterleave.engine import try_deinterleavers
from signalforge.fec.engine import decode_candidates
from signalforge.frame.engine import analyze_frames
from signalforge.validation.validator import validate_chain
from signalforge.observatory.plots import time_domain, spectrum, waterfall, constellation


def analyze_file(path: str, **kwargs) -> dict:
    raw = load_signal(path, **kwargs)
    prepared = prepare(raw)
    observations = observe(prepared)
    params = extract_parameters(prepared)
    modulation = advise(prepared)
    hypotheses = rank_hypotheses(generate(prepared))

    rows = []
    visual = {
        "time_domain": time_domain(prepared.samples, prepared.sample_rate),
        "spectrum": spectrum(prepared.samples, prepared.sample_rate),
        "waterfall": waterfall(prepared.samples, prepared.sample_rate),
    }

    for h in hypotheses[:3]:
        dsp = execute_hypothesis(prepared, h)
        if dsp.get("status") in ("FAILED", "BLOCKED"):
            rows.append({"hypothesis": h.as_dict(), "status": dsp.get("status"), "dsp": dsp})
            continue

        symbols = dsp["symbols"]
        demod = demodulate(prepared, h.modulation, symbols, dsp["integer_samples_per_symbol"])
        visual[f"constellation_{h.modulation}"] = constellation(symbols)

        deints = try_deinterleavers(demod.hard_bits, top_k=3)
        fecs = decode_candidates(demod.hard_bits, demod.soft_bits)
        best_deint = deints[0] if deints else None
        best_fec = fecs[0] if fecs else None
        corrected = best_fec.corrected_bits if best_fec and best_fec.success else (
            best_deint.bits if best_deint else demod.hard_bits
        )
        frames = analyze_frames(corrected)
        frame_score = max((x["score"] for x in frames["candidates"]), default=0.0)
        validation = validate_chain(
            demod.confidence,
            best_deint.score if best_deint else 0.0,
            best_fec.confidence if best_fec else 0.0,
            frame_score,
        )
        rows.append({
            "hypothesis": h.as_dict(),
            "dsp": {k: v for k, v in dsp.items() if k != "symbols"},
            "demodulation": demod.as_dict(),
            "bit_statistics": bit_statistics(demod.hard_bits),
            "deinterleaving_candidates": [x.as_dict() for x in deints],
            "fec_candidates": [x.as_dict() for x in fecs],
            "frame_analysis": frames,
            "validation": validation,
        })

    return {
        "record": prepared.summary(),
        "provenance": prepared.provenance,
        "parameters": {k: v.as_dict() for k, v in params.items()},
        "modulation_candidates": modulation["candidates"],
        "hypotheses": [h.as_dict() for h in hypotheses],
        "executions": rows,
        "visual": visual,
        "observation_keys": list(observations.keys()),
    }


def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    app = tk.Tk()
    app.title("SignalForge V1 — Signal Analysis Observatory")
    app.geometry("1200x760")

    state = {"result": None}

    top = ttk.Frame(app, padding=10)
    top.pack(fill="x")

    path_var = tk.StringVar(value="No file selected")
    status_var = tk.StringVar(value="Ready")

    def choose():
        p = filedialog.askopenfilename(
            title="Select IQ / WAV",
            filetypes=[
                ("Signal files", "*.wav *.iq *.raw *.bin *.cf32 *.fc32"),
                ("WAV", "*.wav"),
                ("All files", "*.*"),
            ],
        )
        if p:
            path_var.set(p)

    def run():
        p = path_var.get()
        if not Path(p).is_file():
            messagebox.showerror("SignalForge", "Select a valid signal file.")
            return
        try:
            status_var.set("Analyzing...")
            app.update_idletasks()
            state["result"] = analyze_file(p)
            r = state["result"]
            record = r["record"]
            params = r["parameters"]
            text.delete("1.0", "end")
            text.insert("end", "SIGNALFORGE V1 ANALYSIS\\n")
            text.insert("end", "=" * 70 + "\\n\\n")
            text.insert("RECORD\\n")
            text.insert("--------\\n")
            text.insert("Format: %s\\n" % record.get("format"))
            text.insert("Sample rate: %s Hz\\n" % record.get("sample_rate"))
            text.insert("Samples: %s\\n\\n" % record.get("num_samples"))
            text.insert("PARAMETERS\\n")
            for k, v in params.items():
                text.insert("  %-22s %s\\n" % (k, v))
            text.insert("\\nMODULATION CANDIDATES\\n")
            for x in r["modulation_candidates"]:
                text.insert("  %s\\n" % x)
            text.insert("\\nHYPOTHESIS RESULTS\\n")
            for row in r["executions"]:
                text.insert("  %s -> %s\\n" % (
                    row["hypothesis"].get("modulation"),
                    row.get("validation", {}).get("status", row.get("status", "UNKNOWN"))
                ))
            status_var.set("Analysis complete")
        except Exception as exc:
            status_var.set("Error")
            messagebox.showerror("SignalForge", str(exc))

    def export():
        if not state["result"]:
            messagebox.showinfo("SignalForge", "Run an analysis first.")
            return
        p = filedialog.asksaveasfilename(
            title="Export analysis JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if p:
            Path(p).write_text(json.dumps(state["result"], default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x), indent=2), encoding="utf-8")
            status_var.set("Exported")

    ttk.Button(top, text="Open IQ/WAV", command=choose).pack(side="left")
    ttk.Entry(top, textvariable=path_var, width=75).pack(side="left", padx=8)
    ttk.Button(top, text="Analyze", command=run).pack(side="left", padx=4)
    ttk.Button(top, text="Export JSON", command=export).pack(side="left", padx=4)

    notebook = ttk.Notebook(app)
    notebook.pack(fill="both", expand=True, padx=10, pady=5)

    tab = ttk.Frame(notebook)
    notebook.add(tab, text="Analysis")

    text = tk.Text(tab, wrap="none", font=("Consolas", 10))
    text.pack(fill="both", expand=True)

    bottom = ttk.Label(app, textvariable=status_var, anchor="w", padding=6)
    bottom.pack(fill="x")

    app.mainloop()


if __name__ == "__main__":
    launch_gui()

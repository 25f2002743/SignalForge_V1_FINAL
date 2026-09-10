from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks

from signalforge.core.parameters import ParameterEstimate
from signalforge.core.signal_record import SignalRecord


def _pe(conf, evidence, status="DERIVED", name="", value=None):
    return ParameterEstimate(name=name, value=value, confidence=conf, evidence=evidence, status=status)


def estimate_noise_floor(psd_db: np.ndarray) -> ParameterEstimate:
    x = np.asarray(psd_db, dtype=float)
    if x.size == 0:
        return _pe(None, [], "UNOBSERVABLE", "noise_floor_db")
    # Robust lower-quantile estimate; deliberately records methodology as evidence.
    value = float(np.percentile(x, 20))
    spread = float(np.std(x))
    confidence = float(np.clip(1.0 - spread / max(abs(value), 1.0), 0.0, 1.0))
    return _pe(confidence, ["Welch PSD lower-quantile (20%) estimate"], name="noise_floor_db", value=value)


def estimate_snr(psd_db: np.ndarray, noise_floor_db: float) -> ParameterEstimate:
    x = np.asarray(psd_db, dtype=float)
    if x.size == 0:
        return _pe(None, [], "UNOBSERVABLE", "snr_db")
    signal_level = float(np.percentile(x, 95))
    snr = max(0.0, signal_level - noise_floor_db)
    confidence = float(np.clip((snr / 20.0), 0.0, 1.0))
    return _pe(confidence, ["PSD 95th-percentile signal level minus estimated noise floor"],
               name="snr_db", value=snr)


def estimate_bandwidth(
    frequency_hz: np.ndarray,
    psd_db: np.ndarray,
    threshold_db_above_noise: float = 6.0,
) -> ParameterEstimate:
    f = np.asarray(frequency_hz)
    p = np.asarray(psd_db)
    if f.size == 0:
        return _pe(None, [], "UNOBSERVABLE", "occupied_bandwidth_hz")

    noise = float(np.percentile(p, 20))
    mask = p >= noise + threshold_db_above_noise
    if not np.any(mask):
        return _pe(0.0, [f"PSD threshold = noise floor + {threshold_db_above_noise:.1f} dB"],
                   "AMBIGUOUS", "occupied_bandwidth_hz", 0.0)

    edges = np.flatnonzero(mask)
    lo_i, hi_i = int(edges[0]), int(edges[-1])
    lo, hi = float(f[lo_i]), float(f[hi_i])
    bw = abs(hi - lo)
    conf = float(np.clip(mask.mean() * 5.0, 0.0, 1.0))
    return _pe(
        conf,
        [f"PSD contiguous threshold region at noise floor + {threshold_db_above_noise:.1f} dB"],
        name="occupied_bandwidth_hz",
        value={"lower_edge_hz": lo, "upper_edge_hz": hi, "bandwidth_hz": bw},
    )


def estimate_peaks(
    frequency_hz: np.ndarray,
    magnitude_db: np.ndarray,
    prominence_db: float = 3.0,
) -> ParameterEstimate:
    f, p = np.asarray(frequency_hz), np.asarray(magnitude_db)
    if f.size < 3:
        return _pe(None, [], "UNOBSERVABLE", "spectral_peaks")
    idx, props = find_peaks(p, prominence=prominence_db)
    order = np.argsort(p[idx])[::-1]
    idx = idx[order]
    top = idx[:50]
    values = [
        {
            "frequency_hz": float(f[i]),
            "power_db": float(p[i]),
            "prominence_db": float(props["prominences"][order][j]),
        }
        for j, i in enumerate(top)
    ]
    return _pe(
        float(np.clip(len(values) / 10.0, 0.0, 1.0)),
        [f"FFT local maxima with prominence >= {prominence_db:.1f} dB"],
        name="spectral_peaks",
        value=values,
    )


def envelope_features(samples: np.ndarray) -> ParameterEstimate:
    a = np.abs(np.asarray(samples))
    if a.size == 0:
        return _pe(None, [], "UNOBSERVABLE", "envelope")
    mean = float(np.mean(a))
    variance = float(np.var(a))
    peak_to_average = float(np.max(a) / max(mean, 1e-15))
    return _pe(
        0.9,
        ["Sample-domain envelope |x[n]|"],
        name="envelope",
        value={
            "mean": mean,
            "variance": variance,
            "peak_to_average": peak_to_average,
            "amplitude_stability": float(1.0 / (1.0 + np.sqrt(variance))),
        },
    )


def phase_features(samples: np.ndarray) -> ParameterEstimate:
    x = np.asarray(samples)
    if not np.iscomplexobj(x):
        return _pe(None, ["Real-only input has no direct complex phase."],
                   "UNOBSERVABLE", "phase")
    phase = np.unwrap(np.angle(x))
    jumps = np.diff(phase)
    return _pe(
        0.9,
        ["Unwrapped angle(x[n])", "Phase-difference statistics"],
        name="phase",
        value={
            "mean_rad": float(np.mean(phase)),
            "std_rad": float(np.std(phase)),
            "median_abs_phase_step_rad": float(np.median(np.abs(jumps))) if jumps.size else 0.0,
            "max_abs_phase_step_rad": float(np.max(np.abs(jumps))) if jumps.size else 0.0,
        },
    )


def instantaneous_frequency(samples: np.ndarray, sample_rate: float) -> ParameterEstimate:
    x = np.asarray(samples)
    if not np.iscomplexobj(x) or x.size < 2:
        return _pe(None, ["Complex I/Q phase is required."], "UNOBSERVABLE", "instantaneous_frequency_hz")
    phase = np.unwrap(np.angle(x))
    fi = np.diff(phase) * sample_rate / (2.0 * np.pi)
    return _pe(
        0.85,
        ["Unwrapped phase derivative: Δφ·Fs/(2π)"],
        name="instantaneous_frequency_hz",
        value={
            "mean_hz": float(np.mean(fi)),
            "median_hz": float(np.median(fi)),
            "std_hz": float(np.std(fi)),
            "p05_hz": float(np.percentile(fi, 5)),
            "p95_hz": float(np.percentile(fi, 95)),
        },
    )


def timing_evidence(samples: np.ndarray, sample_rate: float) -> ParameterEstimate:
    x = np.asarray(samples)
    if x.size < 32:
        return _pe(None, ["Capture too short for robust timing periodicity."],
                   "UNOBSERVABLE", "timing")
    # Squared-envelope autocorrelation is a generic timing evidence source.
    a = np.abs(x) ** 2
    a = a - np.mean(a)
    corr = np.correlate(a, a, mode="full")[a.size - 1:]
    corr[0] = 0
    max_lag = min(len(corr) - 1, max(8, len(corr) // 4))
    idx = int(np.argmax(corr[1:max_lag]) + 1)
    periodicity = float(corr[idx] / max(corr[0:1].max() if corr[0] else np.sum(a*a), 1e-15))
    return _pe(
        float(np.clip(abs(periodicity), 0.0, 1.0)),
        ["Squared-envelope autocorrelation"],
        name="timing",
        value={
            "candidate_period_samples": idx,
            "candidate_symbol_rate_hz": float(sample_rate / idx),
            "timing_strength": periodicity,
        },
    )


def symbol_rate_candidates(
    samples: np.ndarray,
    sample_rate: float,
    top_k: int = 5,
) -> ParameterEstimate:
    x = np.asarray(samples)
    if x.size < 64:
        return _pe(None, [], "UNOBSERVABLE", "symbol_rate_candidates")

    a = np.abs(x) ** 2
    a = a - np.mean(a)
    # FFT of envelope power emphasizes cyclic/timing frequencies.
    n = int(2 ** np.ceil(np.log2(min(x.size, 65536))))
    spec = np.abs(np.fft.rfft(a, n=n))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    if spec.size <= 2:
        return _pe(None, [], "UNOBSERVABLE", "symbol_rate_candidates")
    spec[:2] = 0
    idx, _ = find_peaks(spec)
    if idx.size == 0:
        return _pe(None, ["Envelope-spectrum cyclic-frequency search"],
                   "AMBIGUOUS", "symbol_rate_candidates", [])
    idx = idx[np.argsort(spec[idx])[::-1]][:top_k]
    mx = float(np.max(spec[idx])) if idx.size else 1.0
    candidates = [
        {"symbol_rate_hz": float(freqs[i]),
         "relative_score": float(spec[i] / max(mx, 1e-15))}
        for i in idx
        if freqs[i] > 0
    ]
    return _pe(
        0.65 if candidates else None,
        ["Envelope-power spectral periodicity"],
        "DERIVED" if candidates else "AMBIGUOUS",
        "symbol_rate_candidates",
        candidates,
    )


def frequency_offset_candidates(samples: np.ndarray, sample_rate: float) -> ParameterEstimate:
    x = np.asarray(samples)
    if not np.iscomplexobj(x) or x.size < 8:
        return _pe(None, ["Complex I/Q is required."], "UNOBSERVABLE",
                   "frequency_offset_candidates")
    phase = np.unwrap(np.angle(x))
    inst = np.diff(phase) * sample_rate / (2 * np.pi)
    # Robust central tendency gives a coarse CFO candidate, not a final correction.
    med = float(np.median(inst))
    mad = float(np.median(np.abs(inst - med)))
    confidence = float(np.clip(1.0 / (1.0 + mad / max(abs(med), 1.0)), 0, 1))
    return _pe(
        confidence,
        ["Median instantaneous-frequency estimate", "MAD robustness check"],
        name="frequency_offset_candidates",
        value=[{"offset_hz": med, "relative_confidence": confidence}],
    )


def extract_parameters(record: SignalRecord) -> dict[str, ParameterEstimate]:
    obs = record.observations
    estimates: dict[str, ParameterEstimate] = {}

    estimates["sample_rate"] = _pe(
        1.0, ["Input format / recording metadata"],
        "KNOWN", "sample_rate_hz", float(record.sample_rate)
    )
    estimates["duration"] = _pe(
        1.0, ["N / Fs"],
        "DERIVED", "duration_s", float(record.duration)
    )

    psd = obs.get("psd")
    spec = obs.get("spectrum")
    if psd:
        estimates["noise_floor"] = estimate_noise_floor(psd["psd_db"])
        estimates["snr"] = estimate_snr(psd["psd_db"], estimates["noise_floor"].value)
        estimates["bandwidth"] = estimate_bandwidth(psd["frequency_hz"], psd["psd_db"])
    if spec:
        estimates["spectral_peaks"] = estimate_peaks(
            spec["frequency_hz"], spec["magnitude_db"]
        )

    estimates["envelope"] = envelope_features(record.samples)
    estimates["phase"] = phase_features(record.samples)
    estimates["instantaneous_frequency"] = instantaneous_frequency(
        record.samples, record.sample_rate
    )
    estimates["timing"] = timing_evidence(record.samples, record.sample_rate)
    estimates["symbol_rate_candidates"] = symbol_rate_candidates(
        record.samples, record.sample_rate
    )
    estimates["frequency_offset_candidates"] = frequency_offset_candidates(
        record.samples, record.sample_rate
    )

    record.parameters = {k: v.as_dict() for k, v in estimates.items()}
    record.add_provenance("parameter_extraction", parameter_count=len(estimates))
    return estimates

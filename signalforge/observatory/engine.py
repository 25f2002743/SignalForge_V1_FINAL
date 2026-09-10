from __future__ import annotations

from signalforge.core.signal_record import SignalRecord
from signalforge.observatory.waveform import time_axis, waveform_components
from signalforge.observatory.spectrum import fft_spectrum, spectral_peaks
from signalforge.observatory.psd import welch_psd
from signalforge.observatory.waterfall import waterfall
from signalforge.observatory.constellation import constellation


def observe(
    record: SignalRecord,
    waterfall_fft_size: int = 1024,
) -> dict:
    x = record.samples
    obs = {
        "waveform": {
            "time_s": time_axis(record.num_samples, record.sample_rate),
            **waveform_components(x),
        },
        "spectrum": fft_spectrum(x, record.sample_rate),
        "psd": welch_psd(x, record.sample_rate),
    }

    spec = obs["spectrum"]
    obs["spectrum"]["peaks"] = spectral_peaks(
        spec["frequency_hz"], spec["magnitude_db"]
    )

    if record.num_samples >= waterfall_fft_size:
        obs["waterfall"] = waterfall(
            x, record.sample_rate, fft_size=waterfall_fft_size
        )
    else:
        obs["waterfall"] = None

    if record.is_complex:
        obs["constellation"] = constellation(x)
    else:
        obs["constellation"] = None

    record.observations.update(obs)
    record.add_provenance("signal_observatory")
    return obs

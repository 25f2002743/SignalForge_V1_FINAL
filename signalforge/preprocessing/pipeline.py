from __future__ import annotations
from signalforge.core.signal_record import SignalRecord
from signalforge.preprocessing.basic import remove_dc, normalize


def prepare(record: SignalRecord, normalize_amplitude: bool = True) -> SignalRecord:
    out = remove_dc(record)
    if normalize_amplitude:
        out = normalize(out)
    out.add_provenance("preprocessing_pipeline")
    return out

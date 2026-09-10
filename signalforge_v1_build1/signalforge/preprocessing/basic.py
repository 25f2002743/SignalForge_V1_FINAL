from __future__ import annotations
import numpy as np
from signalforge.core.signal_record import SignalRecord
from signalforge.ingestion.normalizer import amplitude_normalize


def remove_dc(record: SignalRecord) -> SignalRecord:
    derived = record.samples - np.mean(record.samples)
    out = SignalRecord(
        samples=derived,
        sample_rate=record.sample_rate,
        center_frequency=record.center_frequency,
        timestamp=record.timestamp,
        source_format=record.source_format,
        dtype=str(derived.dtype),
        channel_info=dict(record.channel_info),
        metadata=dict(record.metadata),
        provenance=list(record.provenance),
    )
    out.add_provenance("dc_removal")
    return out


def normalize(record: SignalRecord) -> SignalRecord:
    derived = amplitude_normalize(record.samples)
    out = SignalRecord(
        samples=derived,
        sample_rate=record.sample_rate,
        center_frequency=record.center_frequency,
        timestamp=record.timestamp,
        source_format=record.source_format,
        dtype=str(derived.dtype),
        channel_info=dict(record.channel_info),
        metadata=dict(record.metadata),
        provenance=list(record.provenance),
    )
    out.add_provenance("amplitude_normalization")
    return out

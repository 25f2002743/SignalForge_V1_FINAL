from signalforge.pipeline import SignalForgePipeline


def test_pipeline_api_exists():
    p = SignalForgePipeline()
    assert callable(p.analyze)

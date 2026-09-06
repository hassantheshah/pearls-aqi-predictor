from src.feature_pipeline.hazard import fuzzy_hazard


def test_fuzzy_hazard_returns_memberships():
    result = fuzzy_hazard(180)
    assert "dominant" in result
    assert result["memberships"]["Unhealthy"] > 0

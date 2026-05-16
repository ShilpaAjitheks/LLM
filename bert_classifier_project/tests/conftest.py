import pandas as pd
import pytest


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "text": [
            "stocks rallied today after the central bank announcement",
            "the team won the championship in dramatic fashion",
            "earnings reports show strong growth in tech sector",
            "the player scored a hat-trick in the final game",
            "investors are watching the bond market closely",
            "the coach praised the squad after a hard-fought win",
            "the index closed at a record high yesterday",
            "the goalkeeper made several outstanding saves",
            "merger talks have boosted the share price",
            "the tournament will be held next summer",
        ],
        "label_text": [
            "business", "sport", "business", "sport", "business",
            "sport", "business", "sport", "business", "sport",
        ],
    })
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return p

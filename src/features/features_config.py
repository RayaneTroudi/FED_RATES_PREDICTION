def get_feature_config():

    return {
        "VIXCLS": {
            "frequency": "daily",
            "transformations": {
                "ma": {"windows": [7, 14, 63, 126]},  
                "diff": {"lags": [1, 5]},
                "gap": {"windows": [63]}               
            }
        },
        "UNRATE": {
            "frequency": "monthly",
            "transformations": {
                "ma": {"windows": [3, 6]},
                "diff": {"lags": [1]}
            }
        },
        "CPIAUCSL": {
            "frequency": "monthly",
            "transformations": {
                "diff": {"lags": [1]}
            }
        },
        "T10Y2Y": {
            "frequency": "daily",
            "transformations": {
                "diff": {"lags": [1]}
            }
        }
    }


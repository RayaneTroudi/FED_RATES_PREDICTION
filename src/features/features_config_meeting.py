def get_feature_config():

    return {
        "DFF": {
            "frequency": "daily",
            "transformations": {
                "diff": {"lags": [1]}
            }
        },
        "VIXCLS": {
            "frequency": "daily",
            "transformations": {
                "ma": {"windows":[]},  
                "diff": {"lags": [1,2,3,6,12]},
                "gap": {"windows": []}               
            }
        },
        "UNRATE": {
            "frequency": "monthly",
            "transformations": {
                "ma": {"windows": []},
                "diff": {"lags": [1,2,3,6,9,12]}
            }
        },
        "CPIAUCSL": {
            "frequency": "monthly",
            "transformations": {
                "diff": {"lags": [1,2,3,6,12]}
            }
        },
        "T10Y2Y": {
            "frequency": "daily",
            "transformations": {
                "ma" : {"windows":[]},
                "diff": {"lags": [1,2,3,6,12]}
            }
        }
    }


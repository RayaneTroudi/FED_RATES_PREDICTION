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
                "ma": {"windows":[4,8,12]},  
                "diff": {"lags": [1]},
                "gap": {"windows": []}               
            }
        },
        "UNRATE": {
            "frequency": "monthly",
            "transformations": {
                "ma": {"windows": [4,8,12]},
                "diff": {"lags": [1,2]}
            }
        },
        "CPIAUCSL": {
            "frequency": "monthly",
            "transformations": {
                "diff": {"lags": [1,2]}
            }
        },
        "T10Y2Y": {
            "frequency": "daily",
            "transformations": {
                "ma" : {"windows":[4,8,12]},
                "diff": {"lags": [1,2]}
            }
        }
    }


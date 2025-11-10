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
                "ma": {"windows": [28]},  
                "diff": {"lags": [3,6,9,12]},
                "gap": {"windows": []}               
            }
        },
        "UNRATE": {
            "frequency": "monthly",
            "transformations": {
                "ma": {"windows": [3]},
                "diff": {"lags": [3,6,9,12]}
            }
        },
        "CPIAUCSL": {
            "frequency": "monthly",
            "transformations": {
                "diff": {"lags": [3,6,9,12]}
            }
        },
        "T10Y2Y": {
            "frequency": "daily",
            "transformations": {
                "ma" : {"windows":[28]},
                "diff": {"lags": [3,6,9,12]}
            }
        } 
    }
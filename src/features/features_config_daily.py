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
                "ma": {"windows": [7, 14, 21,28]},  
                "diff": {"lags": [1]},
                "gap": {"windows": []}               
            }
        },
        "UNRATE": {
            "frequency": "monthly",
            "transformations": {
                "ma": {"windows": [3, 6,9,12]},
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
                "ma" : {"windows":[7,14,21,28]},
                "diff": {"lags": [1]}
            }
        } 
    }
typeFilm = 'EBT4'
import os
doseCurveJsonPath = os.path.join(os.path.dirname(__file__), "calibration_params_EBTXD.json")
film_labels =   [
    [100, 89, 102, 108, 106, 76, 77],
    [93, 104, 74, 103, 107, 98, 80],
    [81, 94, 92, 73, 101, 83, 88],
    [90, 85, 96, 105, 87, 91, 82],
    [95, 84, 75, 99, 79, 86, 97],
    [54, 49, 71, 72, 56],
    # 
    # [50, 70, 53, 57, 69, 51, 68],
    # [60, 55, 63, 40, 44, 39, 65],
    # [47, 37, 58, 45, 41, 64, 46],
    # [61, 38, 48, 67, 43, 42, 62],
    # 
    # [78, 90, 84, 82, 94, 98, 74],
    # [107, 106, 77, 104, 105, 99, 96],
    # [ 101,91],
]
roiFraction = .5
Directory = "/home/michaelgajda/Downloads/Film/Film"

# 
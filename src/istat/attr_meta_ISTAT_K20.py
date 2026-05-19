"""
attr_meta_ISTAT_K20.py
----------------------
Streamlined 20-attribute structural metadata, domain sizes, anchor 
marginals, and filtered conditional probability tables (CPTs).
"""

import numpy as np

# ------------------------------------------------------------------ #
#  Attribute definitions (K=20)                                      #
# ------------------------------------------------------------------ #

_ATTR_DEFS = [
    ("sex",               ["F", "M"]),
    ("age",               ["0-4", "5-14", "15-24", "25-34", "35-49", "50-64", "65-74", "75+"]),
    ("marital",           ["NeverMarried", "Married", "Divorced", "Widowed"]),
    ("education",         ["SecondaryAndLess", "UpperSecondary", "Tertiary"]),
    ("ResidenceQ",        ["CommuteInward", "Reno", "Navile", "Saragozza", "SanDonato", "SantoStefano", "Savena"]),
    ("StudentStat",       ["NotStudent", "SchoolStudent", "UniStudent"]),
    ("employment",        ["FullTime", "PartTime", "Unemployed", "NotInLF"]),
    ("employ_stat",       ["NotWorker", "SelfEmployed", "Employee"]),
    ("Wage",              ["NotWorker", "Low", "Medium", "High", "VeryHigh"]),
    ("employ_commute",    ["NotWorker", "InsideBO", "Outward", "Inward"]),
    ("Student_commute",   ["NotStudent", "InsideBO", "Outward", "Inward"]),
    ("Occupation",        ["NotWorker", "Manager", "WhiteC", "BlueC", "Elementary"]),
    ("MainTranspStudnt",  ["NotStudent", "Foot", "Bike", "PublicTrns", "CarDriver", "CarPassanger", "MotorCycle"]),
    ("MainTranspWorker",  ["NotWorker", "Foot", "Bike", "PublicTrns", "CarDriver", "CarPassanger", "MotorCycle"]),
    ("TranspTime_Stud",   ["NotStudent", "15m-", "15-30m", "30m+"]),
    ("TranspTime_Worker", ["NotWorker", "15m-", "15-30m", "30m+"]),
    ("LunchPlace",        ["Home", "Canteen", "Restaurant", "Cafe", "AtS/WPlace"]),
    ("SundayOut",         ["Under3yo", "ExitHouse", "StayIn"]),
    ("SaturdayOut",       ["Under3yo", "ExitHouse", "StayIn"]),
    ("WeekDayOut",        ["Under3yo", "ExitHouse", "StayIn"]),
]

ATTR_NAMES_SYNTH   = [name for name, _ in _ATTR_DEFS]
DOMAIN_SIZES_SYNTH = np.array([len(vals) for _, vals in _ATTR_DEFS], dtype=np.int32)

ATTR_META = {
    name: {
        'idx':        idx,
        'vals':       vals,
        'val_to_int': {v: i for i, v in enumerate(vals)},
    }
    for idx, (name, vals) in enumerate(_ATTR_DEFS)
}

K_SYNTH = len(ATTR_NAMES_SYNTH)

# ------------------------------------------------------------------ #
#  Anchor & Implied Marginals                                        #
# ------------------------------------------------------------------ #

marginals = {
    "sex":            {"F": 0.52,  "M": 0.48},
    "age":            {"0-4": 0.03, "5-14": 0.08, "15-24": 0.08, "25-34": 0.13, "35-49": 0.21, "50-64": 0.22, "65-74": 0.11, "75+": 0.14},
    "marital":        {"NeverMarried": 0.51, "Married": 0.38, "Divorced": 0.04, "Widowed": 0.07},
    "StudentStat":    {"NotStudent": 0.72, "SchoolStudent": 0.11, "UniStudent": 0.17},
    "education":      {"SecondaryAndLess": 0.52, "UpperSecondary": 0.34, "Tertiary": 0.14},
    "ResidenceQ":     {"CommuteInward": 0.27, "Reno": 0.11, "Navile": 0.13, "Saragozza": 0.13, "SanDonato": 0.15, "SantoStefano": 0.10, "Savena": 0.11},
    "employment":     {"FullTime": 0.38, "PartTime": 0.08, "Unemployed": 0.02, "NotInLF": 0.52},
    "employ_stat":    {"NotWorker": 0.54, "SelfEmployed": 0.10, "Employee": 0.36},
    "Wage":           {"NotWorker": 0.54, "Low": 0.10, "Medium": 0.17, "High": 0.14, "VeryHigh": 0.05},
    "employ_commute": {"NotWorker": 0.54, "InsideBO": 0.22, "Outward": 0.08, "Inward": 0.16},
    "Student_commute":{"NotStudent": 0.67, "InsideBO": 0.17, "Outward": 0.05, "Inward": 0.11},
    "Occupation":     {"NotWorker": 0.54, "Manager": 0.16, "WhiteC": 0.14, "BlueC": 0.12, "Elementary": 0.04},
    "MainTranspStudnt": {"NotStudent": 0.67, "Foot": 0.03, "Bike": 0.04, "PublicTrns": 0.14, "CarDriver": 0.08, "CarPassanger": 0.03, "MotorCycle": 0.01},
    "MainTranspWorker": {"NotWorker": 0.54, "Foot": 0.02, "Bike": 0.02, "PublicTrns": 0.16, "CarDriver": 0.13, "CarPassanger": 0.10, "MotorCycle": 0.03},
    "TranspTime_Stud" :{"NotStudent": 0.67, "15m-": 0.19, "15-30m": 0.10, "30m+": 0.04},
    "TranspTime_Worker" :{"NotWorker": 0.54, "15m-": 0.19, "15-30m": 0.21, "30m+": 0.06},
    "LunchPlace":     {"Home": 0.73, "Canteen": 0.11, "Restaurant": 0.02, "Cafe": 0.02, "AtS/WPlace": 0.12},
    "SundayOut":      {"Under3yo": 0.02, "ExitHouse": 0.74, "StayIn": 0.24},
    "SaturdayOut":    {"Under3yo": 0.02, "ExitHouse": 0.85, "StayIn": 0.13},
    "WeekDayOut":     {"Under3yo": 0.02, "ExitHouse": 0.86, "StayIn": 0.12},
}

# ------------------------------------------------------------------ #
#  Binary CPTs                                                       #
# ------------------------------------------------------------------ #

age_sex = {
    "0-4":   {"F": 0.55, "M": 0.45}, "5-14":  {"F": 0.49, "M": 0.51},
    "15-24": {"F": 0.50, "M": 0.50}, "25-34": {"F": 0.49, "M": 0.51},
    "35-49": {"F": 0.50, "M": 0.50}, "50-64": {"F": 0.53, "M": 0.47},
    "65-74": {"F": 0.55, "M": 0.45}, "75+":   {"F": 0.61, "M": 0.39},
}

marital_sex = {
    "NeverMarried": {"F": 0.49, "M": 0.51}, "Married": {"F": 0.51, "M": 0.49},
    "Divorced": {"F": 0.64, "M": 0.36}, "Widowed": {"F": 0.81, "M": 0.19},
}

age_marital = {
    "0-4":   {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
    "5-14":  {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
    "15-24": {"NeverMarried": 0.99, "Married": 0.01, "Divorced": 0.00, "Widowed": 0.00},
    "25-34": {"NeverMarried": 0.84, "Married": 0.15, "Divorced": 0.01, "Widowed": 0.00},
    "35-49": {"NeverMarried": 0.50, "Married": 0.46, "Divorced": 0.03, "Widowed": 0.01},
    "50-64": {"NeverMarried": 0.30, "Married": 0.58, "Divorced": 0.09, "Widowed": 0.03},
    "65-74": {"NeverMarried": 0.15, "Married": 0.65, "Divorced": 0.10, "Widowed": 0.10},
    "75+":   {"NeverMarried": 0.07, "Married": 0.49, "Divorced": 0.05, "Widowed": 0.39},
}

sex_education = {
    "F":   {"SecondaryAndLess": 0.50, "UpperSecondary": 0.33, "Tertiary": 0.17},
    "M":   {"SecondaryAndLess": 0.51, "UpperSecondary": 0.35, "Tertiary": 0.14},
}

age_education = {
    "0-4":   {"SecondaryAndLess": 1.00, "UpperSecondary": 0.00, "Tertiary": 0.00},
    "5-14":  {"SecondaryAndLess": 1.00, "UpperSecondary": 0.00, "Tertiary": 0.00},
    "15-24": {"SecondaryAndLess": 0.48, "UpperSecondary": 0.45, "Tertiary": 0.07},
    "25-34": {"SecondaryAndLess": 0.19, "UpperSecondary": 0.49, "Tertiary": 0.32},
    "35-49": {"SecondaryAndLess": 0.25, "UpperSecondary": 0.48, "Tertiary": 0.27},
    "50-64": {"SecondaryAndLess": 0.45, "UpperSecondary": 0.42, "Tertiary": 0.13},
    "65-74": {"SecondaryAndLess": 0.55, "UpperSecondary": 0.34, "Tertiary": 0.11},
    "75+":   {"SecondaryAndLess": 0.73, "UpperSecondary": 0.20, "Tertiary": 0.07},
}

Residence_Sex = {
    "CommuteInward": {"F": 0.48, "M": 0.52}, "Reno": {"F": 0.52, "M": 0.48},
    "Navile": {"F": 0.51, "M": 0.49}, "Saragozza": {"F": 0.53, "M": 0.47},
    "SanDonato": {"F": 0.52, "M": 0.48}, "SantoStefano": {"F": 0.54, "M": 0.46},
    "Savena": {"F": 0.53, "M": 0.47},
}

employment_sex = {
    "FullTime": {"F": 0.38, "M": 0.62}, "PartTime": {"F": 0.78, "M": 0.22},
    "Unemployed": {"F": 0.57, "M": 0.43}, "NotInLF": {"F": 0.56, "M": 0.44},
}

studentStat_sex = {"UniStudent": {"F": 0.57, "M": 0.43}}

studentStat_age = {
    "UniStudent": {"0-4": 0.00, "5-14": 0.00, "15-24": 0.61, "25-34": 0.29, "35-49": 0.10, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

employstat_sex = {"SelfEmployed": {"F": 0.14, "M": 0.86}, "Employee": {"F": 0.25, "M": 0.75}}

occupation_sex = {
    "Manager": {"F": 0.45, "M": 0.55}, "WhiteC": {"F": 0.67, "M": 0.33},
    "BlueC": {"F": 0.17, "M": 0.83}, "Elementary": {"F": 0.48, "M": 0.52},
}

sex_MainTranspWorker = {
    "F": {"NotWorker": 0.60, "Foot": 0.06, "Bike": 0.01, "PublicTrns": 0.03, "CarDriver": 0.27, "CarPassanger": 0.02, "MotorCycle": 0.01},
    "M": {"NotWorker": 0.48, "Foot": 0.05, "Bike": 0.02, "PublicTrns": 0.03, "CarDriver": 0.38, "CarPassanger": 0.02, "MotorCycle": 0.02},
}

sex_TranspTimeWork = {
    "F": {"NotWorker": 0.60, "15m-": 0.16, "15-30m": 0.18, "30m+": 0.06},
    "M": {"NotWorker": 0.48, "15m-": 0.18, "15-30m": 0.26, "30m+": 0.08},
}

sex_LunchPlace = {
    "F": {"Home": 0.83, "Canteen": 0.06, "Restaurant": 0.01, "Cafe": 0.01, "AtS/WPlace": 0.09},
    "M": {"Home": 0.72, "Canteen": 0.08, "Restaurant": 0.05, "Cafe": 0.02, "AtS/WPlace": 0.13},
}

age_LunchPlace = {
    "0-4": {"Home": 1.00, "Canteen": 0.00, "Restaurant": 0.00, "Cafe": 0.00, "AtS/WPlace": 0.00},
    "50-64": {"Home": 0.70, "Canteen": 0.06, "Restaurant": 0.04, "Cafe": 0.03, "AtS/WPlace": 0.17},
    "75+": {"Home": 0.98, "Canteen": 0.00, "Restaurant": 0.01, "Cafe": 0.01, "AtS/WPlace": 0.00},
}

education_LunchPlace = {
    "UpperSecondary": {"Home": 0.73, "Canteen": 0.07, "Restaurant": 0.04, "Cafe": 0.01, "AtS/WPlace": 0.15},
    "Tertiary": {"Home": 0.63, "Canteen": 0.09, "Restaurant": 0.06, "Cafe": 0.03, "AtS/WPlace": 0.19},
}

sex_wage = {
    "F": {"NotWorker": 0.60, "Low": 0.09, "Medium": 0.16, "High": 0.14, "VeryHigh": 0.01},
    "M": {"NotWorker": 0.48, "Low": 0.10, "Medium": 0.19, "High": 0.14, "VeryHigh": 0.09},
}

age_employment = {
    "0-4":   {"FullTime": 0.00, "PartTime": 0.00, "Unemployed": 0.00, "NotInLF": 1.00},
    "5-14":  {"FullTime": 0.00, "PartTime": 0.00, "Unemployed": 0.00, "NotInLF": 1.00},
    "15-24": {"FullTime": 0.22, "PartTime": 0.09, "Unemployed": 0.07, "NotInLF": 0.62},
    "25-34": {"FullTime": 0.86, "PartTime": 0.05, "Unemployed": 0.04, "NotInLF": 0.05},
    "35-49": {"FullTime": 0.70, "PartTime": 0.17, "Unemployed": 0.03, "NotInLF": 0.10},
    "50-64": {"FullTime": 0.57, "PartTime": 0.15, "Unemployed": 0.02, "NotInLF": 0.26},
    "65-74": {"FullTime": 0.10, "PartTime": 0.02, "Unemployed": 0.00, "NotInLF": 0.88},
    "75+":   {"FullTime": 0.00, "PartTime": 0.00, "Unemployed": 0.00, "NotInLF": 1.00},
}

age_wage = {
    "0-4":   {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "5-14":  {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "15-24": {"NotWorker": 0.69, "Low": 0.12, "Medium": 0.12, "High": 0.07, "VeryHigh": 0.00},
    "25-34": {"NotWorker": 0.09, "Low": 0.26, "Medium": 0.35, "High": 0.24, "VeryHigh": 0.06},
    "35-49": {"NotWorker": 0.13, "Low": 0.16, "Medium": 0.32, "High": 0.27, "VeryHigh": 0.12},
    "75+":   {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
}

occupation_wage = {
    "NotWorker":   {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "BlueC":       {"NotWorker": 0.00, "Low": 0.27, "Medium": 0.44, "High": 0.29, "VeryHigh": 0.00},
    "Elementary":  {"NotWorker": 0.00, "Low": 0.51, "Medium": 0.45, "High": 0.04, "VeryHigh": 0.00},
}

education_wage = {
    "SecondaryAndLess": {"NotWorker": 0.61, "Low": 0.10, "Medium": 0.16, "High": 0.13, "VeryHigh": 0.00},
    "UpperSecondary":   {"NotWorker": 0.34, "Low": 0.10, "Medium": 0.26, "High": 0.21, "VeryHigh": 0.09},
    "Tertiary":         {"NotWorker": 0.23, "Low": 0.07, "Medium": 0.26, "High": 0.19, "VeryHigh": 0.25},
}

SundayOut_Sex = {"F": {"Under3yo": 0.02, "ExitHouse": 0.71, "StayIn": 0.27}, "M": {"Under3yo": 0.02, "ExitHouse": 0.82, "StayIn": 0.16}}
SaturdayOut_Sex = {"F": {"Under3yo": 0.02, "ExitHouse": 0.82, "StayIn": 0.16}, "M": {"Under3yo": 0.02, "ExitHouse": 0.88, "StayIn": 0.10}}
WeekDayOut_Sex = {"F": {"Under3yo": 0.02, "ExitHouse": 0.83, "StayIn": 0.15}, "M": {"Under3yo": 0.02, "ExitHouse": 0.91, "StayIn": 0.07}}

EmployCommute_TransTime = {
    "NotWorker": {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "InsideBO":  {"NotWorker": 0.00, "15m-": 0.65, "15-30m": 0.27, "30m+": 0.08},
    "Outward":   {"NotWorker": 0.00, "15m-": 0.23, "15-30m": 0.22, "30m+": 0.55},
    "Inward":    {"NotWorker": 0.00, "15m-": 0.05, "15-30m": 0.18, "30m+": 0.77},
}

StudentCommute_TransTime = {
    "NotStudent": {"NotStudent": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "InsideBO":   {"NotStudent": 0.00, "15m-": 0.44, "15-30m": 0.41, "30m+": 0.15},
    "Outward":    {"NotStudent": 0.00, "15m-": 0.18, "15-30m": 0.43, "30m+": 0.39},
    "Inward":     {"NotStudent": 0.00, "15m-": 0.03, "15-30m": 0.20, "30m+": 0.77},
}

EmployCommute_MainTransp = {
    "NotWorker": {"NotWorker": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "InsideBO":  {"NotWorker": 0.00, "Foot": 0.06, "Bike": 0.07, "PublicTrns": 0.20, "CarDriver": 0.23, "CarPassanger": 0.36, "MotorCycle": 0.08},
    "Outward":   {"NotWorker": 0.00, "Foot": 0.00, "Bike": 0.01, "PublicTrns": 0.37, "CarDriver": 0.48, "CarPassanger": 0.12, "MotorCycle": 0.03},
    "Inward":    {"NotWorker": 0.00, "Foot": 0.00, "Bike": 0.01, "PublicTrns": 0.54, "CarDriver": 0.28, "CarPassanger": 0.15, "MotorCycle": 0.03},
}

StudentCommute_MainTransp = {
    "NotStudent": {"NotStudent": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "InsideBO":   {"NotStudent": 0.00, "Foot": 0.16, "Bike": 0.24, "PublicTrns": 0.52, "CarDriver": 0.02, "CarPassanger": 0.03, "MotorCycle": 0.03},
    "Outward":    {"NotStudent": 0.00, "Foot": 0.00, "Bike": 0.01, "PublicTrns": 0.26, "CarDriver": 0.55, "CarPassanger": 0.14, "MotorCycle": 0.04},
    "Inward":     {"NotStudent": 0.00, "Foot": 0.00, "Bike": 0.01, "PublicTrns": 0.35, "CarDriver": 0.46, "CarPassanger": 0.14, "MotorCycle": 0.04},
}

MainTranspStudnt_TranspTimeS = {
    "NotStudent":   {"NotStudent": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "Foot":         {"NotStudent": 0.00, "15m-": 0.72, "15-30m": 0.22, "30m+": 0.06},
    "Bike":         {"NotStudent": 0.00, "15m-": 0.53, "15-30m": 0.39, "30m+": 0.08},
    "PublicTrns":   {"NotStudent": 0.00, "15m-": 0.08, "15-30m": 0.33, "30m+": 0.59},
    "CarDriver":    {"NotStudent": 0.00, "15m-": 0.24, "15-30m": 0.44, "30m+": 0.32},
    "CarPassanger": {"NotStudent": 0.00, "15m-": 0.24, "15-30m": 0.42, "30m+": 0.34},
    "MotorCycle":   {"NotStudent": 0.00, "15m-": 0.51, "15-30m": 0.41, "30m+": 0.08},
}

MainTranspWorker_TranspTimeW = {
    "NotWorker":    {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "Foot":         {"NotWorker": 0.00, "15m-": 0.85, "15-30m": 0.13, "30m+": 0.02},
    "Bike":         {"NotWorker": 0.00, "15m-": 0.71, "15-30m": 0.25, "30m+": 0.04},
    "PublicTrns":   {"NotWorker": 0.00, "15m-": 0.11, "15-30m": 0.28, "30m+": 0.61},
    "CarDriver":    {"NotWorker": 0.00, "15m-": 0.10, "15-30m": 0.34, "30m+": 0.56},
    "CarPassanger": {"NotWorker": 0.00, "15m-": 0.67, "15-30m": 0.23, "30m+": 0.10},
    "MotorCycle":   {"NotWorker": 0.00, "15m-": 0.58, "15-30m": 0.36, "30m+": 0.06},
}

# ------------------------------------------------------------------ #
#  Ternary CPTs                                                      #
# ------------------------------------------------------------------ #

marital_age_sex = {
    "0-4": {
        "F": {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
        "M": {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
    },
    "5-14": {
        "F": {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
        "M": {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
    },
    "15-24": {
        "F": {"NeverMarried": 0.98, "Married": 0.02, "Divorced": 0.00, "Widowed": 0.00},
        "M": {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
    },
    "25-34": {
        "F": {"NeverMarried": 0.80, "Married": 0.19, "Divorced": 0.01, "Widowed": 0.00},
        "M": {"NeverMarried": 0.88, "Married": 0.11, "Divorced": 0.01, "Widowed": 0.00},
    },
    "35-49": {
        "F": {"NeverMarried": 0.47, "Married": 0.48, "Divorced": 0.04, "Widowed": 0.01},
        "M": {"NeverMarried": 0.54, "Married": 0.43, "Divorced": 0.02, "Widowed": 0.01},
    },
    "50-64": {
        "F": {"NeverMarried": 0.30, "Married": 0.54, "Divorced": 0.11, "Widowed": 0.05},
        "M": {"NeverMarried": 0.31, "Married": 0.58, "Divorced": 0.07, "Widowed": 0.04},
    },
    "65-74": {
        "F": {"NeverMarried": 0.16, "Married": 0.58, "Divorced": 0.11, "Widowed": 0.15},
        "M": {"NeverMarried": 0.14, "Married": 0.73, "Divorced": 0.08, "Widowed": 0.05},
    },
    "75+": {
        "F": {"NeverMarried": 0.08, "Married": 0.35, "Divorced": 0.05, "Widowed": 0.52},
        "M": {"NeverMarried": 0.07, "Married": 0.71, "Divorced": 0.04, "Widowed": 0.18},
    },
}

sex_age_studentStat = {
    "UniStudent": {
        "15-24":     {"F": 0.58, "M": 0.42},
        "25-34":     {"F": 0.55, "M": 0.45},
        "35-49":     {"F": 0.58, "M": 0.42},
    },
}

employment_age_sex = {
    "FullTime": {
        "35-49":     {"F": 0.36, "M": 0.64},
        "50-64":     {"F": 0.37, "M": 0.63},
    },
    "PartTime": {
        "35-49":     {"F": 0.88, "M": 0.12},
        "50-64":     {"F": 0.84, "M": 0.16},
    },
}

# ------------------------------------------------------------------ #
#  Structural Zeros (Impossible Combinations)                        #
# ------------------------------------------------------------------ #

h_age_marital = {
    "0-4":    {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
    "5-14":   {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
}

h_age_education = {
    "0-4":    {"SecondaryAndLess": 1.00, "UpperSecondary": 0.00, "Tertiary": 0.00},
    "5-14":   {"SecondaryAndLess": 1.00, "UpperSecondary": 0.00, "Tertiary": 0.00},
}

h_EmployCommute_ResidenceQ = {
    "Inward":    {"CommuteInward": 1.00, "Reno": 0.00, "Navile": 0.00, "Saragozza": 0.00, "SanDonato": 0.00, "SantoStefano": 0.00, "Savena": 0.00},
}

h_StudentCommute_ResidenceQ = {
    "Inward":    {"CommuteInward": 1.00, "Reno": 0.00, "Navile": 0.00, "Saragozza": 0.00, "SanDonato": 0.00, "SantoStefano": 0.00, "Savena": 0.00},
}

h_StudentStat_StudentCommute = {
    "NotStudent":    {"NotStudent": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
    "SchoolStudent": {"NotStudent": 0.00, "InsideBO": 0.33, "Outward": 0.33, "Inward": 0.34},
    "UniStudent":    {"NotStudent": 0.00, "InsideBO": 0.33, "Outward": 0.33, "Inward": 0.34},
}

h_StudentStat_MainTranspStudnt = {
    "NotStudent":    {"NotStudent": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "SchoolStudent": {"NotStudent": 0.00, "Foot": 0.16, "Bike": 0.16, "PublicTrns": 0.17, "CarDriver": 0.17, "CarPassanger": 0.17, "MotorCycle": 0.17},
    "UniStudent":    {"NotStudent": 0.00, "Foot": 0.16, "Bike": 0.16, "PublicTrns": 0.17, "CarDriver": 0.17, "CarPassanger": 0.17, "MotorCycle": 0.17},
}

h_StudentStat_TranspTimeStud = {
    "NotStudent":    {"NotStudent": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "SchoolStudent": {"NotStudent": 0.00, "15m-": 0.33, "15-30m": 0.33, "30m+": 0.34},
    "UniStudent":    {"NotStudent": 0.00, "15m-": 0.33, "15-30m": 0.33, "30m+": 0.34},
}

h_employstat_wage = {
    "NotWorker":    {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "SelfEmployed": {"NotWorker": 0.00, "Low": 0.25, "Medium": 0.25, "High": 0.25, "VeryHigh": 0.25},
    "Employee":     {"NotWorker": 0.00, "Low": 0.25, "Medium": 0.25, "High": 0.25, "VeryHigh": 0.25},
}

h_employstat_employcommute = {
    "NotWorker":    {"NotWorker": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
    "SelfEmployed": {"NotWorker": 0.00, "InsideBO": 0.33, "Outward": 0.33, "Inward": 0.34},
    "Employee":     {"NotWorker": 0.00, "InsideBO": 0.33, "Outward": 0.33, "Inward": 0.34},
}

h_employstat_Occupation = {
    "NotWorker":    {"NotWorker": 1.00, "Manager": 0.00, "WhiteC": 0.00, "BlueC": 0.00, "Elementary": 0.00},
    "SelfEmployed": {"NotWorker": 0.00, "Manager": 0.25, "WhiteC": 0.25, "BlueC": 0.25, "Elementary": 0.25},
    "Employee":     {"NotWorker": 0.00, "Manager": 0.25, "WhiteC": 0.25, "BlueC": 0.25, "Elementary": 0.25},
}

h_employstat_MainTranspWorker = {
    "NotWorker":    {"NotWorker": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "SelfEmployed": {"NotWorker": 0.00, "Foot": 0.16, "Bike": 0.16, "PublicTrns": 0.17, "CarDriver": 0.17, "CarPassanger": 0.17, "MotorCycle": 0.17},
    "Employee":     {"NotWorker": 0.00, "Foot": 0.16, "Bike": 0.16, "PublicTrns": 0.17, "CarDriver": 0.17, "CarPassanger": 0.17, "MotorCycle": 0.17},
}

h_employstat_TranspTimeWorker = {
    "NotWorker":    {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "SelfEmployed": {"NotWorker": 0.00, "15m-": 0.33, "15-30m": 0.33, "30m+": 0.34},
    "Employee":     {"NotWorker": 0.00, "15m-": 0.33, "15-30m": 0.33, "30m+": 0.34},
}

h_employment_employstat = {
    "Unemployed": {"NotWorker": 1.00, "SelfEmployed": 0.00, "Employee": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "SelfEmployed": 0.00, "Employee": 0.00},
    "FullTime":   {"NotWorker": 0.00, "SelfEmployed": 0.50, "Employee": 0.50},
    "PartTime":   {"NotWorker": 0.00, "SelfEmployed": 0.50, "Employee": 0.50},
}

h_employment_wage = {
    "Unemployed": {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "FullTime":   {"NotWorker": 0.00, "Low": 0.25, "Medium": 0.25, "High": 0.25, "VeryHigh": 0.25},
    "PartTime":   {"NotWorker": 0.00, "Low": 0.25, "Medium": 0.25, "High": 0.25, "VeryHigh": 0.25},
}

h_employment_employcommute = {
    "Unemployed": {"NotWorker": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
    "FullTime":   {"NotWorker": 0.00, "InsideBO": 0.33, "Outward": 0.33, "Inward": 0.34},
    "PartTime":   {"NotWorker": 0.00, "InsideBO": 0.33, "Outward": 0.33, "Inward": 0.34},
}

h_employment__Occupation = {
    "Unemployed": {"NotWorker": 1.00, "Manager": 0.00, "WhiteC": 0.00, "BlueC": 0.00, "Elementary": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "Manager": 0.00, "WhiteC": 0.00, "BlueC": 0.00, "Elementary": 0.00},
    "FullTime":   {"NotWorker": 0.00, "Manager": 0.25, "WhiteC": 0.25, "BlueC": 0.25, "Elementary": 0.25},
    "PartTime":   {"NotWorker": 0.00, "Manager": 0.25, "WhiteC": 0.25, "BlueC": 0.25, "Elementary": 0.25},
}

h_employment_MainTranspWorker = {
    "Unemployed": {"NotWorker": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "FullTime":   {"NotWorker": 0.00, "Foot": 0.16, "Bike": 0.16, "PublicTrns": 0.17, "CarDriver": 0.17, "CarPassanger": 0.17, "MotorCycle": 0.17},
    "PartTime":   {"NotWorker": 0.00, "Foot": 0.16, "Bike": 0.16, "PublicTrns": 0.17, "CarDriver": 0.17, "CarPassanger": 0.17, "MotorCycle": 0.17},
}

h_employment__TranspTimeWorker = {
    "Unemployed": {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "FullTime":   {"NotWorker": 0.00, "15m-": 0.33, "15-30m": 0.33, "30m+": 0.34},
    "PartTime":   {"NotWorker": 0.00, "15m-": 0.33, "15-30m": 0.33, "30m+": 0.34},
}

h_SundayOut_age = {"Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00}}
h_SaturdayOut_age = {"Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00}}
h_WeekDayOut_age = {"Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00}}

h_age_employment = {
    "0-4":  {"FullTime": 0.00, "PartTime": 0.00, "Unemployed": 0.00, "NotInLF": 1.00},
    "5-14": {"FullTime": 0.00, "PartTime": 0.00, "Unemployed": 0.00, "NotInLF": 1.00},
}

h_age_employstat = {
    "0-4":  {"NotWorker": 1.00, "SelfEmployed": 0.00, "Employee": 0.00},
    "5-14": {"NotWorker": 1.00, "SelfEmployed": 0.00, "Employee": 0.00},
}

h_age_Wage = {
    "0-4":  {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "5-14": {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
}

h_age_StudentStat = {
    "0-4":   {"NotStudent": 1.00, "SchoolStudent": 0.00, "UniStudent": 0.00},
    "50-64": {"NotStudent": 1.00, "SchoolStudent": 0.00, "UniStudent": 0.00},
    "65-74": {"NotStudent": 1.00, "SchoolStudent": 0.00, "UniStudent": 0.00},
    "75+":   {"NotStudent": 1.00, "SchoolStudent": 0.00, "UniStudent": 0.00},
}

h_employment_StudentStat = {
    "FullTime": {"NotStudent": 1.00, "SchoolStudent": 0.00, "UniStudent": 0.00},
}

h_employcommute_studentcommute = {
    "Inward":   {"NotStudent": 0.93, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.07},
    "Outward":  {"NotStudent": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
}

h_MainTranspStudnt_MainTranspWorker = {
    "Foot":         {"NotWorker": 0.85, "Foot": 0.15, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "Bike":         {"NotWorker": 0.85, "Foot": 0.00, "Bike": 0.15, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "PublicTrns":   {"NotWorker": 0.85, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.15, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "CarDriver":    {"NotWorker": 0.85, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.15, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "CarPassanger": {"NotWorker": 0.85, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.15, "MotorCycle": 0.00},
    "MotorCycle":   {"NotWorker": 0.85, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.15},
}
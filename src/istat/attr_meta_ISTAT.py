"""
attr_meta_ISTAT.py
------------
Attribute metadata, domain sizes, anchor marginals and
conditional probability tables (CPTs) for the ISTAT benchmark.

"""

import numpy as np

# ------------------------------------------------------------------ #
#  Attribute definitions                                               #
# ------------------------------------------------------------------ #

_ATTR_DEFS = [
    ("sex",               ["F", "M"]),
    ("age",               ["0-4", "5-14", "15-24", "25-34", "35-49", "50-64", "65-74", "75+"]),
    ("marital",           ["NeverMarried", "Married", "Divorced", "Widowed"]),
    ("citizenship",       ["Italian", "Foreigner"]), #We can easlily add continent/country also permessotypeandpass var
    ("education",         ["SecondaryAndLess", "UpperSecondary", "Tertiary"]),
    ("ResidenceQ",        ["CommuteInward", "Reno", "Navile", "Saragozza", "SanDonato", "SantoStefano", "Savena"]), #ExtraAssumption W/S commuter independancy (almost verified by checking ISTAT 2011 questionaire)
    ("StudentStat",       ["NotStudent", "SchoolStudent", "UniStudent"]),
    ("employment",        ["FullTime", "PartTime", "Unemployed", "NotInLF"]),
    ("employ_stat",       ["NotWorker", "SelfEmployed", "Employee"]),
    ("Wage",              ["NotWorker", "Low", "Medium", "High", "VeryHigh"]),
    ("employ_commute",    ["NotWorker", "InsideBO", "Outward", "Inward"]), 
    ("Student_commute",   ["NotStudent", "InsideBO", "Outward", "Inward"]), 
    ("Profession",        ["NotWorker", "AgricFishForest", "Industry", "Services"]),
    ("Occupation",        ["NotWorker", "Manager", "WhiteC", "BlueC", "Elementary"]),
    ("BMI",               ["UnderAge", "UnderWeight", "NormalWeight", "OverWeight", "Obese"]),
    ("Health",            ["Healthy", "ChronicGoodHealth", "ChronicBadHealth"]),
    ("Medication",        ["NotOnMed", "OnMed"]),
    ("AlcoholCons",       ["Never", "Exceptionally", "Consumer"]),
    ("Smoking",           ["Never", "Former", "1-5", "6-10", "11-20", "20+"]),
    ("MainTranspStudnt",  ["NotStudent", "Foot", "Bike", "PublicTrns", "CarDriver", "CarPassanger", "MotorCycle"]),
    ("MainTranspWorker",  ["NotWorker", "Foot", "Bike", "PublicTrns", "CarDriver", "CarPassanger", "MotorCycle"]),
    ("TranspTime_Stud",   ["NotStudent", "15m-", "15-30m", "30m+"]), 
    ("TranspTime_Worker", ["NotWorker", "15m-", "15-30m", "30m+"]), 
    ("LunchPlace",        ["Home", "Canteen", "Restaurant", "Cafe", "AtS/WPlace"]),
    ("SundayOut",         ["Under3yo", "ExitHouse", "StayIn"]),
    ("SaturdayOut",       ["Under3yo", "ExitHouse", "StayIn"]),
    ("WeekDayOut",        ["Under3yo", "ExitHouse", "StayIn"]),
    ("SunSocialEnterT",   ["Under3yo", "Y", "N"]),
    ("SatSocialEnterT",   ["Under3yo", "Y", "N"]),
    ("WeekDSocialEnterT", ["Under3yo", "Y", "N"]),
    ("SunSportOutD",      ["Under3yo", "Y", "N"]),
    ("SatSportOutD",      ["Under3yo", "Y", "N"]),
    ("WeekDSportOutD",    ["Under3yo", "Y", "N"]),
    ("LifeSatisfaction",  ["Under14yo", "0-3", "4-6", "7-10"]),
    
]

### HH, residing area,,

ATTR_NAMES_SYNTH   = [name for name, _ in _ATTR_DEFS]
DOMAIN_SIZES_SYNTH = np.array([len(vals) for _, vals in _ATTR_DEFS],
                               dtype=np.int32)

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
#  Anchor marginals                                                     #
# ------------------------------------------------------------------ #

marginals = {
    "sex":            {"F": 0.52,  "M": 0.48}, #BO
    "age":            {"0-4": 0.03, "5-14": 0.08, "15-24": 0.08, #BO
                       "25-34": 0.13, "35-49": 0.21, "50-64": 0.22, "65-74": 0.11, "75+": 0.14}, #BO
    "marital":        {"NeverMarried": 0.51, "Married": 0.38, "Divorced": 0.04, "Widowed": 0.07}, #BO
    "StudentStat":    {"NotStudent": 0.72, "SchoolStudent": 0.11, "UniStudent": 0.17}, #BO
    "BMI":            {"UnderAge": 0.11, "UnderWeight": 0.03, "NormalWeight": 0.42, "OverWeight": 0.33, "Obese": 0.11}, #EmiliaR
    "Health":         {"Healthy": 0.57, "ChronicGoodHealth": 0.12, "ChronicBadHealth": 0.31}, #EmiliaR
    "Medication":     {"NotOnMed": 0.55, "OnMed": 0.45}, #EmiliaR
    # implied marginals (derived from binary/ternary tables)
    "citizenship":    {"Italian": 0.85, "Foreigner": 0.15}, #BO
    "education":      {"SecondaryAndLess": 0.52, "UpperSecondary": 0.34, "Tertiary": 0.14}, #Northeast
    "ResidenceQ":     {"CommuteInward": 0.27, "Reno": 0.11, "Navile": 0.13, "Saragozza": 0.13, "SanDonato": 0.15, "SantoStefano": 0.10, "Savena": 0.11}, #BO 2023 extraAssumpCommuter
    "employment":     {"FullTime": 0.38, "PartTime": 0.08, "Unemployed": 0.02, "NotInLF": 0.52}, #EmiliaR
    "employ_stat":    {"NotWorker": 0.54, "SelfEmployed": 0.10, "Employee": 0.36}, #PBO
    "Wage":           {"NotWorker": 0.54, "Low": 0.10, "Medium": 0.17, "High": 0.14, "VeryHigh": 0.05}, #PBO 2023
    "employ_commute": {"NotWorker": 0.54, "InsideBO": 0.22, "Outward": 0.08, "Inward": 0.16}, #BO 2023
    "Student_commute":{"NotStudent": 0.67, "InsideBO": 0.17, "Outward": 0.05, "Inward": 0.11}, #BO 2011
    "Profession":     {"NotWorker": 0.54, "AgricFishForest": 0.01, "Industry": 0.12, "Services": 0.33}, #BO
    "Occupation":     {"NotWorker": 0.54, "Manager": 0.16, "WhiteC": 0.14, "BlueC": 0.12, "Elementary": 0.04}, #NorthEast
    "AlcoholCons":    {"Never": 0.59, "Exceptionally": 0.20, "Consumer": 0.21}, #EmiliaR
    "Smoking":        {"Never": 0.57, "Former": 0.24, "1-5": 0.05, "6-10": 0.07, "11-20": 0.06, "20+": 0.01}, #EmiliaR
    "MainTranspStudnt": {"NotStudent": 0.67, "Foot": 0.03, "Bike": 0.04, "PublicTrns": 0.14, "CarDriver": 0.08, "CarPassanger": 0.03, "MotorCycle": 0.01}, #EmiliaR 2023
    "MainTranspWorker": {"NotWorker": 0.54, "Foot": 0.02, "Bike": 0.02, "PublicTrns": 0.16, "CarDriver": 0.13, "CarPassanger": 0.10, "MotorCycle": 0.03}, #Emilia 2023
    "TranspTime_Stud" :{"NotStudent": 0.67, "15m-": 0.19, "15-30m": 0.10, "30m+": 0.04}, #EmiliaR 2023
    "TranspTime_Worker" :{"NotWorker": 0.54, "15m-": 0.19, "15-30m": 0.21, "30m+": 0.06}, #EmiliaR 2023
    "LunchPlace":     {"Home": 0.73, "Canteen": 0.11, "Restaurant": 0.02, "Cafe": 0.02, "AtS/WPlace": 0.12}, #EmiliaR 2023
    "SundayOut":      {"Under3yo": 0.02, "ExitHouse": 0.74, "StayIn": 0.24}, #EmiliaR 2013
    "SaturdayOut":    {"Under3yo": 0.02, "ExitHouse": 0.85, "StayIn": 0.13}, #EmiliaR 2013
    "WeekDayOut":     {"Under3yo": 0.02, "ExitHouse": 0.86, "StayIn": 0.12}, #EmiliaR 2013
    "SunSocialEnterT":{"Under3yo": 0.02, "Y": 0.72, "N": 0.26}, #EmiliaR 2013
    "SatSocialEnterT":{"Under3yo": 0.02, "Y": 0.75, "N": 0.23}, #EmiliaR 2013
    "WeekDSocialEnterT":{"Under3yo": 0.02, "Y": 0.72, "N": 0.26}, #EmiliaR 2013
    "SunSportOutD":   {"Under3yo": 0.02, "Y": 0.42, "N": 0.56}, #EmiliaR 2013
    "SatSportOutD":   {"Under3yo": 0.02, "Y": 0.37, "N": 0.61}, #EmiliaR 2013
    "WeekDSportOutD": {"Under3yo": 0.02, "Y": 0.32, "N": 0.66}, #EmiliaR 2013
    "LifeSatisfaction":{"Under14yo": 0.11, "0-3": 0.03, "4-6": 0.20, "7-10": 0.66}, #EmiliaR 2023
    
}


### We have lots of undownloaded data for activity(Rate) CPT with couples and HH
# ------------------------------------------------------------------ #
#  Binary CPTs                                                          #
# ------------------------------------------------------------------ #



####Conditions: 
# B1: P(sex | age) #BO
age_sex = {
    "0-4":   {"F": 0.55, "M": 0.45},
    "5-14":  {"F": 0.49, "M": 0.51},
    "15-24": {"F": 0.50, "M": 0.50},
    "25-34": {"F": 0.49, "M": 0.51},
    "35-49": {"F": 0.50, "M": 0.50},
    "50-64": {"F": 0.53, "M": 0.47},
    "65-74": {"F": 0.55, "M": 0.45},
    "75+":   {"F": 0.61, "M": 0.39},
}

# B2: P(sex | marital) #BO
marital_sex = {
    "NeverMarried":  {"F": 0.49, "M": 0.51},
    "Married": {"F": 0.51, "M": 0.49},
    "Divorced": {"F": 0.64, "M": 0.36},
    "Widowed":   {"F": 0.81, "M": 0.19},
}

# B3: P(marital | age) #BO
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

# B4: P(Sex | Citizenship) #BO
citizenship_sex = {
    "Italian":   {"F": 0.52, "M": 0.48},
    "Foreigner": {"F": 0.52, "M": 0.48},

}

# B5: P(Education | Sex) #NorthEast
sex_education = {
    "F":   {"SecondaryAndLess": 0.50, "UpperSecondary": 0.33, "Tertiary": 0.17},
    "M":   {"SecondaryAndLess": 0.51, "UpperSecondary": 0.35, "Tertiary": 0.14},
}

# B6: P(Education | Age) #NorthEast
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

# B7: P(Education | citizenship) #NorthEast
citizenship_education = {
    "Italian":   {"SecondaryAndLess": 0.51, "UpperSecondary": 0.35, "Tertiary": 0.14},
    "Foreigner": {"SecondaryAndLess": 0.57, "UpperSecondary": 0.33, "Tertiary": 0.10},
}

# B8: P(Sex | ResidenceQ) #BO
Residence_Sex = {
    "CommuteInward":   {"F": 0.48, "M": 0.52},
    "Reno": {"F": 0.52, "M": 0.48},
    "Navile":   {"F": 0.51, "M": 0.49},
    "Saragozza": {"F": 0.53, "M": 0.47},
    "SanDonato":   {"F": 0.52, "M": 0.48},
    "SantoStefano": {"F": 0.54, "M": 0.46},
    "Savena":   {"F": 0.53, "M": 0.47},
}


# B9: P(Sex | Employment) #EmiliaR Labour Force and EmiliaR PartTimeFullTime
employment_sex = {
    "FullTime":   {"F": 0.38, "M": 0.62},
    "PartTime": {"F": 0.78, "M": 0.22},
    "Unemployed":   {"F": 0.57, "M": 0.43},
    "NotInLF": {"F": 0.56, "M": 0.44},
}


# B10: P(Sex | StudentStat) #BO
studentStat_sex = {
    "UniStudent":   {"F": 0.57, "M": 0.43},
}

# B11: P(Age | StudentStat) #EmiliaR
studentStat_age = {
    "UniStudent":    {"0-4": 0.00, "5-14": 0.00, "15-24": 0.61, "25-34": 0.34, "35-49": 0.05, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

 
# B12: P(Sex | employ_stat) #NorthEast
employstat_sex = {
    "SelfEmployed":    {"F": 0.14, "M": 0.86},
    "Employee":        {"F": 0.25, "M": 0.75},
    #NotWorker Absent
}


#B13: P(Citizenship | employment) #NorthEast
employment_citizenship = {
    "FullTime":      {"Italian": 0.88, "Foreigner": 0.12},
    "PartTime":      {"Italian": 0.88, "Foreigner": 0.12},
    # Absent
}


#B14: P(Sex | Profession) #IT
profession_sex = {
    "AgricFishForest":    {"F": 0.25, "M": 0.75},
    "Industry":        {"F": 0.26, "M": 0.74},
    "Services":        {"F": 0.55, "M": 0.45},
    # Absent
}


#B15: P(employment | Profession) #IT
profession_employment = {
    "AgricFishForest":    {"FullTime": 0.85, "PartTime": 0.15},
    "Industry":        {"FullTime": 0.92, "PartTime": 0.08},
    "Services":        {"FullTime": 0.76, "PartTime": 0.24},
    # Absent
}

#B16: P(employstat | Profession) #IT
profession_employstat = {
    "AgricFishForest":    {"SelfEmployed": 0.58, "Employee": 0.42},
    "Industry":        {"SelfEmployed": 0.14, "Employee": 0.86},
    "Services":        {"SelfEmployed": 0.22, "Employee": 0.78},
    # Absent
}

#B17: P(citizenship | employstat) #NorthEast
employstat_citizenship = {
    "SelfEmployed":    {"Italian": 0.93, "Foreigner": 0.07},
    "Employee":        {"Italian": 0.87, "Foreigner": 0.13},
    # Absent
}

#B18 P(Citizenship | profession) #NorthEast
profession_citizenship = {
    "AgricFishForest": {"Italian": 0.85, "Foreigner": 0.15},
    "Industry":        {"Italian": 0.87, "Foreigner": 0.13},
    "Services":        {"Italian": 0.89, "Foreigner": 0.11},
}


#B19 P(Sex | Occupation) #NorthEast
occupation_sex = {
    "Manager":      {"F": 0.45, "M": 0.55},
    "WhiteC":       {"F": 0.67, "M": 0.33},
    "BlueC":        {"F": 0.17, "M": 0.83},
    "Elementary":   {"F": 0.48, "M": 0.52},
}

#B20 P(Citizenship | Occupation) #NorthEast
occupation_citizenship = {
    "Manager":      {"Italian": 0.97, "Foreigner": 0.03},
    "WhiteC":       {"Italian": 0.88, "Foreigner": 0.12},
    "BlueC":        {"Italian": 0.83, "Foreigner": 0.17},
    "Elementary":   {"Italian": 0.67, "Foreigner": 0.33},
}

#B21 P(BMI | Sex) #IT
sex_BMI = {
    "M":      {"UnderAge": 0.09, "UnderWeight": 0.01, "NormalWeight": 0.40, "OverWeight": 0.38, "Obese": 0.12},
    "F":      {"UnderAge": 0.13, "UnderWeight": 0.05, "NormalWeight": 0.49, "OverWeight": 0.23, "Obese": 0.10},
}

#B22 P(Health | sex) #IT
sex_health = {
    "M":      {"Healthy": 0.54, "ChronicGoodHealth": 0.19, "ChronicBadHealth": 0.27},
    "F":      {"Healthy": 0.49, "ChronicGoodHealth": 0.16, "ChronicBadHealth": 0.35},
}

#B23 P(Health | Age) #IT
age_health = {
    "5-14":  {"Healthy": 0.89, "ChronicGoodHealth": 0.06, "ChronicBadHealth": 0.05},
    "15-24": {"Healthy": 0.76, "ChronicGoodHealth": 0.13, "ChronicBadHealth": 0.11},
    "25-34": {"Healthy": 0.73, "ChronicGoodHealth": 0.13, "ChronicBadHealth": 0.14},
    "35-49": {"Healthy": 0.67, "ChronicGoodHealth": 0.14, "ChronicBadHealth": 0.19},
    "50-64": {"Healthy": 0.34, "ChronicGoodHealth": 0.25, "ChronicBadHealth": 0.41},
    "65-74": {"Healthy": 0.19, "ChronicGoodHealth": 0.27, "ChronicBadHealth": 0.54},
    "75+":   {"Healthy": 0.09, "ChronicGoodHealth": 0.20, "ChronicBadHealth": 0.71},
}

#B24 P(Health | Occupation) #IT
occupation_health = {
    "Manager":      {"Healthy": 0.57, "ChronicGoodHealth": 0.21, "ChronicBadHealth": 0.22},
    "WhiteC":       {"Healthy": 0.57, "ChronicGoodHealth": 0.19, "ChronicBadHealth": 0.24},
}

#B25 P(Medication | Sex) #IT
sex_medication = {
    "F":       {"NotOnMed": 0.54, "OnMed": 0.46},
    "M":       {"NotOnMed": 0.61, "OnMed": 0.39},
}

#B26 P(Medication | Age) #IT
age_medication = {
    "15-24": {"NotOnMed": 0.81, "OnMed": 0.19},
    "25-34": {"NotOnMed": 0.78, "OnMed": 0.22},
    "35-49": {"NotOnMed": 0.70, "OnMed": 0.30},
    "50-64": {"NotOnMed": 0.45, "OnMed": 0.55},
    "65-74": {"NotOnMed": 0.25, "OnMed": 0.75},
    "75+":   {"NotOnMed": 0.12, "OnMed": 0.88},
}

#B27 P(Sex | Alcohol) #IT
alcohol_sex = {
    "F":      {"Never": 0.69, "Exceptionally": 0.17, "Consumer": 0.14},
    "M":      {"Never": 0.55, "Exceptionally": 0.22, "Consumer": 0.23},
}

#B28 P(Sex | Smoking) #IT
smoking_sex = {
    "F":      {"Never": 0.68, "Former": 0.17, "1-5": 0.05, "6-10": 0.06, "11-20": 0.04, "20+": 0.00},
    "M":      {"Never": 0.52, "Former": 0.27, "1-5": 0.05, "6-10": 0.07, "11-20": 0.08, "20+": 0.01},
}

#B29 P(WorkMeanTransp | Sex) #IT
sex_MainTranspWorker = {
    "F":      {"NotWorker": 0.60, "Foot": 0.06, "Bike": 0.01, "PublicTrns": 0.03, "CarDriver": 0.27, "CarPassanger": 0.02, "MotorCycle": 0.01},
    "M":      {"NotWorker": 0.48, "Foot": 0.05, "Bike": 0.02, "PublicTrns": 0.03, "CarDriver": 0.38, "CarPassanger": 0.02, "MotorCycle": 0.02},
}

#B30 P(TimeToWork | Sex) #IT
sex_TranspTimeWork = {
    "F":      {"NotWorker": 0.60, "15m-": 0.16, "15-30m": 0.18, "30m+": 0.06},
    "M":      {"NotWorker": 0.48, "15m-": 0.18, "15-30m": 0.26, "30m+": 0.08},
}


#B31 P(LunchPlace | Sex) #IT
sex_LunchPlace = {
    "F":      {"Home": 0.83, "Canteen": 0.06, "Restaurant": 0.01, "Cafe": 0.01, "AtS/WPlace": 0.09},
    "M":      {"Home": 0.72, "Canteen": 0.08, "Restaurant": 0.05, "Cafe": 0.02, "AtS/WPlace": 0.13},
}

#B32 P(LunchPlace | Age) #IT
age_LunchPlace = {
    "0-4":   {"Home": 1.00, "Canteen": 0.00, "Restaurant": 0.00, "Cafe": 0.00, "AtS/WPlace": 0.00},
    "50-64": {"Home": 0.70, "Canteen": 0.06, "Restaurant": 0.04, "Cafe": 0.03, "AtS/WPlace": 0.17},
    "75+":   {"Home": 0.98, "Canteen": 0.00, "Restaurant": 0.01, "Cafe": 0.01, "AtS/WPlace": 0.00},
}

#B33 P(LunchPlace | Education) #IT
education_LunchPlace = {
    "UpperSecondary":   {"Home": 0.73, "Canteen": 0.07, "Restaurant": 0.04, "Cafe": 0.01, "AtS/WPlace": 0.15},
    "Tertiary":         {"Home": 0.63, "Canteen": 0.09, "Restaurant": 0.06, "Cafe": 0.03, "AtS/WPlace": 0.19},
}

#B34 P(Wage | Sex) #PBO
sex_wage = {
    "F":     {"NotWorker": 0.60, "Low": 0.09, "Medium": 0.16, "High": 0.14, "VeryHigh": 0.01},
    "M":     {"NotWorker": 0.48, "Low": 0.10, "Medium": 0.19, "High": 0.14, "VeryHigh": 0.09},
}


#B35 P(Employment | age) #NorthEast
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


#B36 P(wage | Age) #PBO
age_wage = {
    "0-4":   {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "5-14":  {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "15-24": {"NotWorker": 0.69, "Low": 0.12, "Medium": 0.12, "High": 0.07, "VeryHigh": 0.00},
    "25-34": {"NotWorker": 0.09, "Low": 0.26, "Medium": 0.35, "High": 0.24, "VeryHigh": 0.06},
    "35-49": {"NotWorker": 0.13, "Low": 0.16, "Medium": 0.32, "High": 0.27, "VeryHigh": 0.12},
    "75+":   {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
}

#B37 P(wage | citizenship) #PBO
citizenship_wage = {
    "Italian":    {"NotWorker": 0.55, "Low": 0.06, "Medium": 0.18, "High": 0.12, "VeryHigh": 0.09},
    "Foreigner":  {"NotWorker": 0.50, "Low": 0.17, "Medium": 0.22, "High": 0.11, "VeryHigh": 0.00},
}

#B38 P(Wage | Occupation) #PBO
occupation_wage = {
    "NotWorker":   {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "BlueC":       {"NotWorker": 0.00, "Low": 0.27, "Medium": 0.44, "High": 0.29, "VeryHigh": 0.00},
    "Elementary":  {"NotWorker": 0.00, "Low": 0.51, "Medium": 0.45, "High": 0.04, "VeryHigh": 0.00},
}

#B39 P(Wage | Education) #PBO
education_wage = {
    "SecondaryAndLess":   {"NotWorker": 0.61, "Low": 0.10, "Medium": 0.16, "High": 0.13, "VeryHigh": 0.00},
    "UpperSecondary":     {"NotWorker": 0.34, "Low": 0.10, "Medium": 0.26, "High": 0.21, "VeryHigh": 0.09},
    "Tertiary":           {"NotWorker": 0.23, "Low": 0.07, "Medium": 0.26, "High": 0.19, "VeryHigh": 0.25},
}

#B40 P(Sex | SundayOut) #IT
SundayOut_Sex = {
    "F":      {"Under3yo": 0.02, "ExitHouse": 0.71, "StayIn": 0.27},
    "M":      {"Under3yo": 0.02, "ExitHouse": 0.82, "StayIn": 0.16},
}

#B41 P(Sex | SaturdayOut) #IT
SaturdayOut_Sex = {
    "F":      {"Under3yo": 0.02, "ExitHouse": 0.82, "StayIn": 0.16},
    "M":      {"Under3yo": 0.02, "ExitHouse": 0.88, "StayIn": 0.10},
}

#B42 P(Sex | WeekDayOut) #IT
WeekDayOut_Sex = {
    "F":      {"Under3yo": 0.02, "ExitHouse": 0.83, "StayIn": 0.15},
    "M":      {"Under3yo": 0.02, "ExitHouse": 0.91, "StayIn": 0.07},
}

#B43 P(Sex | SunSocialEnterT) #IT
SunSocialEnterT_Sex = {
    "F":      {"Under3yo": 0.02, "Y": 0.56, "N": 0.42},
    "M":      {"Under3yo": 0.02, "Y": 0.59, "N": 0.39},
}

#B44 P(Sex | SatSocialEnterT) #IT
SatSocialEnterT_Sex = {
    "F":      {"Under3yo": 0.02, "Y": 0.56, "N": 0.42},
    "M":      {"Under3yo": 0.02, "Y": 0.60, "N": 0.38},
}

#B45 P(Sex | WeekDSocialEnterT) #IT
WeekDSocialEnterT_Sex = {
    "F":      {"Under3yo": 0.02, "Y": 0.54, "N": 0.44},
    "M":      {"Under3yo": 0.02, "Y": 0.50, "N": 0.48},
}

#B46 P(Sex | SunSportOutD) #IT
SunSportOutD_Sex = {
    "F":      {"Under3yo": 0.02, "Y": 0.38, "N": 0.60},
    "M":      {"Under3yo": 0.02, "Y": 0.47, "N": 0.51},
}

#B47 P(Sex | SatSportOutD) #IT
SatSportOutD_Sex = {
    "F":      {"Under3yo": 0.02, "Y": 0.31, "N": 0.67},
    "M":      {"Under3yo": 0.02, "Y": 0.40, "N": 0.58},
}

#B48 P(Sex | WeekDSportOutD) #IT
WeekDSportOutD_Sex = {
    "F":      {"Under3yo": 0.02, "Y": 0.26, "N": 0.72},
    "M":      {"Under3yo": 0.02, "Y": 0.32, "N": 0.66},
}

#B49 P(LifeSatisfaction | Sex) #IT
Sex_LifeSatisfaction = {
    "F":      {"Under14yo": 0.11, "0-3": 0.03, "4-6": 0.22, "7-10": 0.64},
    "M":      {"Under14yo": 0.12, "0-3": 0.02, "4-6": 0.20, "7-10": 0.66},
}

#B50 P(LifeSatisfaction | Age) #IT
Age_LifeSatisfaction = {
    "0-4":   {"Under14yo": 1.00, "0-3": 0.00, "4-6": 0.00, "7-10": 0.00},
    "5-14":  {"Under14yo": 1.00, "0-3": 0.00, "4-6": 0.00, "7-10": 0.00},
    "15-24": {"Under14yo": 0.00, "0-3": 0.02, "4-6": 0.19, "7-10": 0.79},
    "25-34": {"Under14yo": 0.00, "0-3": 0.03, "4-6": 0.20, "7-10": 0.77},
    "65-74": {"Under14yo": 0.00, "0-3": 0.02, "4-6": 0.25, "7-10": 0.73},
    "75+":   {"Under14yo": 0.00, "0-3": 0.05, "4-6": 0.30, "7-10": 0.65},
}

#B51 P (LifeSatisfaction | Occupation) IT
Occupation_LifeSatisfaction = {
    "Manager":     {"Under14yo": 0.00, "0-3": 0.01, "4-6": 0.16, "7-10": 0.83},
    "WhiteC":      {"Under14yo": 0.00, "0-3": 0.01, "4-6": 0.19, "7-10": 0.80},
}

#B52 P (LifeSatisfaction | Student) #IT
studentstat_LifeSatisfaction = {
    "UniStudent":     {"Under14yo": 0.00, "0-3": 0.02, "4-6": 0.19, "7-10": 0.79},
}

#B53 P (LifeSatisfaction | Education) #IT
education_LifeSatisfaction = {
    "UpperSecondary":     {"Under14yo": 0.00, "0-3": 0.03, "4-6": 0.22, "7-10": 0.75},
    "Tertiary":           {"Under14yo": 0.00, "0-3": 0.02, "4-6": 0.19, "7-10": 0.79},
}


#B54 P (TranspTime | employCommute) #EmiliaR
EmployCommute_TransTime = {
    "NotWorker": {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "InsideBO":  {"NotWorker": 0.00, "15m-": 0.65, "15-30m": 0.27, "30m+": 0.08},
    "Outward":   {"NotWorker": 0.00, "15m-": 0.23, "15-30m": 0.22, "30m+": 0.55},
    "Inward":    {"NotWorker": 0.00, "15m-": 0.05, "15-30m": 0.18, "30m+": 0.77},
}

#B55 P (TransTime | StudentCommute) #EmiliaR
StudentCommute_TransTime = {
    "NotStudent": {"NotStudent": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "InsideBO":   {"NotStudent": 0.00, "15m-": 0.44, "15-30m": 0.41, "30m+": 0.15},
    "Outward":    {"NotStudent": 0.00, "15m-": 0.18, "15-30m": 0.43, "30m+": 0.39},
    "Inward":     {"NotStudent": 0.00, "15m-": 0.03, "15-30m": 0.20, "30m+": 0.77},
}

#B56 P (MainTranspWorker | EmployCommute) #EmiliaR
EmployCommute_MainTransp = {
    "NotWorker": {"NotWorker": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "InsideBO":   {"NotWorker": 0.00, "Foot": 0.06, "Bike": 0.07, "PublicTrns": 0.20, "CarDriver": 0.23, "CarPassanger": 0.36, "MotorCycle": 0.08},
    "Outward":    {"NotWorker": 0.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.37, "CarDriver": 0.48, "CarPassanger": 0.12, "MotorCycle": 0.03},
    "Inward":     {"NotWorker": 0.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.54, "CarDriver": 0.28, "CarPassanger": 0.15, "MotorCycle": 0.03},
}


#B57 P (MainTranspStud | StudentCommute) #EmiliaR
StudentCommute_MainTransp = {
    "NotStudent": {"NotStudent": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "InsideBO":   {"NotStudent": 0.00, "Foot": 0.16, "Bike": 0.24, "PublicTrns": 0.52, "CarDriver": 0.02, "CarPassanger": 0.03, "MotorCycle": 0.03},
    "Outward":    {"NotStudent": 0.00, "Foot": 0.00, "Bike": 0.01, "PublicTrns": 0.26, "CarDriver": 0.55, "CarPassanger": 0.14, "MotorCycle": 0.04},
    "Inward":     {"NotStudent": 0.00, "Foot": 0.00, "Bike": 0.01, "PublicTrns": 0.35, "CarDriver": 0.46, "CarPassanger": 0.14, "MotorCycle": 0.04},
}

#B58 P (TransTimeStud | MainTransStud) #EmiliaR
MainTranspStudnt_TranspTimeS = {
    "NotStudent":    {"NotStudent": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "Foot":          {"NotStudent": 0.00, "15m-": 0.72, "15-30m": 0.22, "30m+": 0.06},
    "Bike":          {"NotStudent": 0.00, "15m-": 0.53, "15-30m": 0.39, "30m+": 0.08},
    "PublicTrns":    {"NotStudent": 0.00, "15m-": 0.08, "15-30m": 0.33, "30m+": 0.59},
    "CarDriver":     {"NotStudent": 0.00, "15m-": 0.24, "15-30m": 0.44, "30m+": 0.32},
    "CarPassanger":  {"NotStudent": 0.00, "15m-": 0.24, "15-30m": 0.42, "30m+": 0.34},
    "MotorCycle":    {"NotStudent": 0.00, "15m-": 0.51, "15-30m": 0.41, "30m+": 0.08},
}

#B59 P (TransTimeWork | MainTranspWorker) #EmiliaR
MainTranspWorker_TranspTimeW = {
    "NotWorker":     {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "Foot":          {"NotWorker": 0.00, "15m-": 0.85, "15-30m": 0.13, "30m+": 0.02},
    "Bike":          {"NotWorker": 0.00, "15m-": 0.71, "15-30m": 0.25, "30m+": 0.04},
    "PublicTrns":    {"NotWorker": 0.00, "15m-": 0.11, "15-30m": 0.28, "30m+": 0.61},
    "CarDriver":     {"NotWorker": 0.00, "15m-": 0.10, "15-30m": 0.34, "30m+": 0.56},
    "CarPassanger":  {"NotWorker": 0.00, "15m-": 0.67, "15-30m": 0.23, "30m+": 0.10},
    "MotorCycle":    {"NotWorker": 0.00, "15m-": 0.58, "15-30m": 0.36, "30m+": 0.06},
}



# ------------------------------------------------------------------ #
#  Ternary CPTs                                                         #
# ------------------------------------------------------------------ #

# T1: P(marital | age, sex) #BO
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

# T2: P(education | citizenship, sex) #NorthEast
education_sex_citizenship = {
    "Italian": {
        "F":     {"SecondaryAndLess": 0.52, "UpperSecondary": 0.32, "Tertiary": 0.16},
        "M": {"SecondaryAndLess": 0.50, "UpperSecondary": 0.37, "Tertiary": 0.13},
    },
    "Foreigner": {
        "F":     {"SecondaryAndLess": 0.54, "UpperSecondary": 0.34, "Tertiary": 0.12},
        "M": {"SecondaryAndLess": 0.60, "UpperSecondary": 0.33, "Tertiary": 0.07},
    },

}

# T3: P(Sex | StudentStat, age) #IT
sex_age_studentStat = {
    "UniStudent": {
        "15-24":     {"F": 0.58, "M": 0.42},
        "25-34":     {"F": 0.55, "M": 0.45},
        "35-49":     {"F": 0.58, "M": 0.42},
    },
}

#T4: employment age #NorthEast fulltimeParttime Fultime; age; sex 

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


#T5: employment sex citizenship #NorthEast
employment_citizenship_sex = {
    "FullTime": {
        "Italian":     {"F": 0.37, "M": 0.63},
        "Foreigner":     {"F": 0.34, "M": 0.66},
    },
    "PartTime": {
        "Italian":     {"F": 0.80, "M": 0.20},
        "Foreigner":     {"F": 0.83, "M": 0.17},
    },
}


#T6: employstat profession sex #NorthEast
profession_employstat_sex = {
    "AgricFishForest": {
        "Employee":     {"F": 0.29, "M": 0.71},
        "SelfEmployed":     {"F": 0.21, "M": 0.79},
    },
    "Industry": {
        "Employee":     {"F": 0.28, "M": 0.72},
        "SelfEmployed":     {"F": 0.12, "M": 0.88},
    },
    "Services": {
        "Employee":     {"F": 0.59, "M": 0.41},
        "SelfEmployed":     {"F": 0.39, "M": 0.61},
    },
}

#T7: #NorthEast
profession_employment_sex = {
    "AgricFishForest": {
        "FullTime":     {"F": 0.20, "M": 0.80},
        "PartTime":     {"F": 0.52, "M": 0.48},
    },
    "Industry": {
        "FullTime":     {"F": 0.21, "M": 0.79},
        "PartTime":     {"F": 0.77, "M": 0.23},
    },
    "Services": {
        "FullTime":     {"F": 0.46, "M": 0.54},
        "PartTime":     {"F": 0.81, "M": 0.19},
    },
}


#T8: #NorthEast
Profession_employment_employstat = {
    "AgricFishForest": {
        "FullTime":     {"Employee": 0.41, "SelfEmployed": 0.59},
        "PartTime":     {"Employee": 0.44, "SelfEmployed": 0.56},
    },
    "Industry": {
        "FullTime":     {"Employee": 0.86, "SelfEmployed": 0.14},
        "PartTime":     {"Employee": 0.85, "SelfEmployed": 0.15},
    },
    "Services": {
        "FullTime":     {"Employee": 0.76, "SelfEmployed": 0.24},
        "PartTime":     {"Employee": 0.85, "SelfEmployed": 0.15},
    },
}

#T9: #NorthEast
employstat_citizenship_sex = {
    "SelfEmployed": {
        "Italian":     {"F": 0.31, "M": 0.69},
        "Foreigner":     {"F": 0.35, "M": 0.65},
    },
    "Employee": {
        "Italian":     {"F": 0.48, "M": 0.52},
        "Foreigner":     {"F": 0.44, "M": 0.56},
    },
}


#T10 #NorthEast
profession_citizenship_sex = {
    "AgricFishForest": {
        "Italian":     {"F": 0.25, "M": 0.75},
        "Foreigner":   {"F": 0.21, "M": 0.79},
    },
    "Industry": {
        "Italian":     {"F": 0.27, "M": 0.73},
        "Foreigner":   {"F": 0.18, "M": 0.82},
    },
    "Services": {
        "Italian":     {"F": 0.54, "M": 0.46},
        "Foreigner":   {"F": 0.60, "M": 0.40},
    },
}

#T11: #NorthEast
profession_employstat_citizenship = {
    "AgricFishForest": {
        "SelfEmployed":   {"Italian": 0.99, "Foreigner": 0.01},
        "Employee":       {"Italian": 0.67, "Foreigner": 0.33},
    },
    "Industry": {
        "SelfEmployed":   {"Italian": 0.90, "Foreigner": 0.10},
        "Employee":       {"Italian": 0.86, "Foreigner": 0.14},
    },
    "Services": {
        "SelfEmployed":   {"Italian": 0.93, "Foreigner": 0.07},
        "Employee":       {"Italian": 0.88, "Foreigner": 0.12},
    },
}

#T12: #NorthEast
occupation_citizenship_sex = {
    "Manager": {
        "Italian":         {"F": 0.45, "M": 0.55},
        "Foreigner":       {"F": 0.45, "M": 0.55},
    },
    "WhiteC": {
        "Italian":         {"F": 0.66, "M": 0.34},
        "Foreigner":       {"F": 0.73, "M": 0.27},
    },
    "BlueC": {
        "Italian":         {"F": 0.16, "M": 0.84},
        "Foreigner":       {"F": 0.17, "M": 0.83},
    },
    "Elementary": {
        "Italian":         {"F": 0.50, "M": 0.50},
        "Foreigner":       {"F": 0.45, "M": 0.55},
    },
}


#T13: #IT
sex_age_BMI = {
    "M": {
        "0-4":         {"UnderAge": 1.00},
        "5-14":        {"UnderAge": 1.00},
        "15-24":       {"UnderWeight": 0.05, "NormalWeight": 0.72, "OverWeight": 0.18, "Obese": 0.05},
        "25-34":       {"UnderWeight": 0.01, "NormalWeight": 0.58, "OverWeight": 0.32, "Obese": 0.09},
        "35-49":       {"UnderWeight": 0.01, "NormalWeight": 0.48, "OverWeight": 0.40, "Obese": 0.11},
        "50-64":       {"UnderWeight": 0.01, "NormalWeight": 0.36, "OverWeight": 0.47, "Obese": 0.16},
        "65-74":       {"UnderWeight": 0.00, "NormalWeight": 0.32, "OverWeight": 0.51, "Obese": 0.17},
        "75+":         {"UnderWeight": 0.01, "NormalWeight": 0.38, "OverWeight": 0.47, "Obese": 0.14},
    },
    "F": {
        "0-4":         {"UnderAge": 1.00},
        "5-14":        {"UnderAge": 1.00},
        "15-24":       {"UnderWeight": 0.15, "NormalWeight": 0.68, "OverWeight": 0.13, "Obese": 0.04},
        "25-34":       {"UnderWeight": 0.10, "NormalWeight": 0.65, "OverWeight": 0.18, "Obese": 0.07},
        "35-49":       {"UnderWeight": 0.06, "NormalWeight": 0.64, "OverWeight": 0.22, "Obese": 0.08},
        "50-64":       {"UnderWeight": 0.04, "NormalWeight": 0.57, "OverWeight": 0.29, "Obese": 0.10},
        "65-74":       {"UnderWeight": 0.04, "NormalWeight": 0.46, "OverWeight": 0.35, "Obese": 0.15},
        "75+":         {"UnderWeight": 0.04, "NormalWeight": 0.45, "OverWeight": 0.37, "Obese": 0.14},
    },
}


#T14 #IT
occupation_sex_BMI = {
    "Manager": {
        "M":       {"UnderWeight": 0.00, "NormalWeight": 0.44, "OverWeight": 0.42, "Obese": 0.14},
        "F":       {"UnderWeight": 0.07, "NormalWeight": 0.67, "OverWeight": 0.20, "Obese": 0.06},
    },
    "WhiteC": {
        "M":       {"UnderWeight": 0.01, "NormalWeight": 0.48, "OverWeight": 0.40, "Obese": 0.11},
        "F":       {"UnderWeight": 0.07, "NormalWeight": 0.66, "OverWeight": 0.20, "Obese": 0.07},
    },
}

#T15 #IT
StudentStat_Sex_BMI = {
    "UniStudent": {
        "M":       {"UnderWeight": 0.04, "NormalWeight": 0.75, "OverWeight": 0.17, "Obese": 0.04},
        "F":       {"UnderWeight": 0.16, "NormalWeight": 0.67, "OverWeight": 0.13, "Obese": 0.04},
    },
}

#T16 #IT
sex_age_alcohol = {
    "M": {
        "0-4":   {"Never": 1.00, "Exceptionally": 0.00, "Consumer": 0.00},
        "5-14":  {"Never": 0.99, "Exceptionally": 0.01, "Consumer": 0.00},
        "15-24": {"Never": 0.51, "Exceptionally": 0.19, "Consumer": 0.30},
        "25-34": {"Never": 0.31, "Exceptionally": 0.26, "Consumer": 0.43},
        "35-49": {"Never": 0.35, "Exceptionally": 0.27, "Consumer": 0.38},
        "50-64": {"Never": 0.51, "Exceptionally": 0.28, "Consumer": 0.21},
        "65-74": {"Never": 0.65, "Exceptionally": 0.23, "Consumer": 0.12},
        "75+":   {"Never": 0.79, "Exceptionally": 0.14, "Consumer": 0.07},
    },
    "F": {
        "0-4":   {"Never": 1.00, "Exceptionally": 0.00, "Consumer": 0.00},
        "5-14":  {"Never": 1.00, "Exceptionally": 0.00, "Consumer": 0.00},
        "15-24": {"Never": 0.53, "Exceptionally": 0.20, "Consumer": 0.27},
        "25-34": {"Never": 0.43, "Exceptionally": 0.24, "Consumer": 0.33},
        "35-49": {"Never": 0.48, "Exceptionally": 0.27, "Consumer": 0.25},
        "50-64": {"Never": 0.70, "Exceptionally": 0.21, "Consumer": 0.09},
        "65-74": {"Never": 0.84, "Exceptionally": 0.12, "Consumer": 0.04},
        "75+":   {"Never": 0.92, "Exceptionally": 0.06, "Consumer": 0.02},
    },
}

#T17 #IT
profession_sex_alcohol = {
    "Manager": {
        "F":   {"Never": 0.43, "Exceptionally": 0.30, "Consumer": 0.27},
        "M":   {"Never": 0.30, "Exceptionally": 0.33, "Consumer": 0.37},
    },
    "WhiteC": {
        "F":   {"Never": 0.44, "Exceptionally": 0.30, "Consumer": 0.26},
        "M":   {"Never": 0.43, "Exceptionally": 0.25, "Consumer": 0.32},
    },
}


#T18 #IT
age_sex_smoking = {
    "M": {
        "0-4":   {"Never": 1.00, "Former": 0.00, "1-5": 0.00, "6-10": 0.00, "11-20": 0.00, "20+": 0.00},
        "5-14":  {"Never": 1.00, "Former": 0.00, "1-5": 0.00, "6-10": 0.00, "11-20": 0.00, "20+": 0.00},
        "15-24": {"Never": 0.70, "Former": 0.08, "1-5": 0.09, "6-10": 0.08, "11-20": 0.05, "20+": 0.00},
        "25-34": {"Never": 0.49, "Former": 0.16, "1-5": 0.10, "6-10": 0.12, "11-20": 0.12, "20+": 0.01},
        "35-49": {"Never": 0.44, "Former": 0.25, "1-5": 0.08, "6-10": 0.10, "11-20": 0.12, "20+": 0.01},
        "50-64": {"Never": 0.43, "Former": 0.35, "1-5": 0.03, "6-10": 0.07, "11-20": 0.10, "20+": 0.02},
        "65-74": {"Never": 0.35, "Former": 0.46, "1-5": 0.04, "6-10": 0.06, "11-20": 0.08, "20+": 0.01},
        "75+":   {"Never": 0.38, "Former": 0.54, "1-5": 0.03, "6-10": 0.03, "11-20": 0.02, "20+": 0.00},
    },
    "F": {
        "0-4":   {"Never": 1.00, "Former": 0.00, "1-5": 0.00, "6-10": 0.00, "11-20": 0.00, "20+": 0.00},
        "5-14":  {"Never": 1.00, "Former": 0.00, "1-5": 0.00, "6-10": 0.00, "11-20": 0.00, "20+": 0.00},
        "15-24": {"Never": 0.76, "Former": 0.07, "1-5": 0.09, "6-10": 0.05, "11-20": 0.03, "20+": 0.00},
        "25-34": {"Never": 0.63, "Former": 0.17, "1-5": 0.07, "6-10": 0.10, "11-20": 0.03, "20+": 0.00},
        "35-49": {"Never": 0.59, "Former": 0.21, "1-5": 0.06, "6-10": 0.09, "11-20": 0.05, "20+": 0.00},
        "50-64": {"Never": 0.55, "Former": 0.25, "1-5": 0.04, "6-10": 0.08, "11-20": 0.07, "20+": 0.01},
        "65-74": {"Never": 0.59, "Former": 0.27, "1-5": 0.04, "6-10": 0.05, "11-20": 0.05, "20+": 0.00},
        "75+":   {"Never": 0.76, "Former": 0.19, "1-5": 0.02, "6-10": 0.02, "11-20": 0.01, "20+": 0.00},
    },
}

#T19 #IT
profession_sex_MainTranspWorker = {
    "Manager": {
        "F":   {"NotWorker": 0.60, "Foot": 0.07, "Bike": 0.01, "PublicTrns": 0.08, "CarDriver": 0.18, "CarPassanger": 0.05, "MotorCycle": 0.01},
        "M":   {"NotWorker": 0.48, "Foot": 0.10, "Bike": 0.01, "PublicTrns": 0.06, "CarDriver": 0.26, "CarPassanger": 0.06, "MotorCycle": 0.03},
    },
    "WhiteC": {
        "F":   {"NotWorker": 0.60, "Foot": 0.05, "Bike": 0.01, "PublicTrns": 0.07, "CarDriver": 0.19, "CarPassanger": 0.07, "MotorCycle": 0.01},
        "M":   {"NotWorker": 0.48, "Foot": 0.04, "Bike": 0.02, "PublicTrns": 0.07, "CarDriver": 0.29, "CarPassanger": 0.06, "MotorCycle": 0.04},
    },
}


#T20 #IT
profession_sex_TranspTimeWork = {
    "Manager": {
        "F":   {"NotWorker": 0.60, "15m-": 0.15, "15-30m": 0.21, "30m+": 0.04},
        "M":   {"NotWorker": 0.48, "15m-": 0.19, "15-30m": 0.27, "30m+": 0.06},
    },
    "WhiteC": {
        "F":   {"NotWorker": 0.60, "15m-": 0.15, "15-30m": 0.18, "30m+": 0.07},
        "M":   {"NotWorker": 0.48, "15m-": 0.14, "15-30m": 0.23, "30m+": 0.15},
    },
}


# ------------------------------------------------------------------ #
#  Impossible Combinations                                           #
# ------------------------------------------------------------------ #

#H1 #BO
h_age_marital = {
    "0-4":    {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
    "5-14":   {"NeverMarried": 1.00, "Married": 0.00, "Divorced": 0.00, "Widowed": 0.00},
}


#H2 #BO
h_age_education = {
    "0-4":    {"SecondaryAndLess": 1.00, "UpperSecondary": 0.00, "Tertiary": 0.00},
    "5-14":   {"SecondaryAndLess": 1.00, "UpperSecondary": 0.00, "Tertiary": 0.00},
}

#H3 #BO
h_EmployCommute_ResidenceQ = {
    "Inward":    {"CommuteInward": 1.00, "Reno": 0.00, "Navile": 0.00, "Saragozza": 0.00, "SanDonato": 0.00, "SantoStefano": 0.00, "Savena": 0.00},
}

#H4 #BO
h_StudentCommute_ResidenceQ = {
    "Inward":    {"CommuteInward": 1.00, "Reno": 0.00, "Navile": 0.00, "Saragozza": 0.00, "SanDonato": 0.00, "SantoStefano": 0.00, "Savena": 0.00},
}

# ------------------------------------------------------------------ #
#  STRUCTURAL ZERO-MAPPING: STUDENTS (H5 - H7)
# ------------------------------------------------------------------ #

#H5 BO
h_StudentStat_StudentCommute = {
    "NotStudent":    {"NotStudent": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
    "SchoolStudent": {"NotStudent": 0.00},
    "UniStudent":    {"NotStudent": 0.00},
}


#H6 #BO
h_StudentStat_MainTranspStudnt = {
    "NotStudent":    {"NotStudent": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "SchoolStudent": {"NotStudent": 0.00},
    "UniStudent":    {"NotStudent": 0.00},
}

#H7 #BO
h_StudentStat_TranspTimeStud = {
    "NotStudent":    {"NotStudent": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "SchoolStudent": {"NotStudent": 0.00},
    "UniStudent":    {"NotStudent": 0.00},
}

# ------------------------------------------------------------------ #
#  STRUCTURAL ZERO-MAPPING: EMPLOY_STAT (H8 - H13)
# ------------------------------------------------------------------ #
#H8 #BO
h_employstat_wage = {
    "NotWorker":    {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "SelfEmployed": {"NotWorker": 0.00},
    "Employee":     {"NotWorker": 0.00},
}

#H9 #BO
h_employstat_employcommute = {
    "NotWorker":    {"NotWorker": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
    "SelfEmployed": {"NotWorker": 0.00},
    "Employee":     {"NotWorker": 0.00},
}

#H10 #BO
h_employstat_Profession = {
    "NotWorker":    {"NotWorker": 1.00, "AgricFishForest": 0.00, "Industry": 0.00, "Services": 0.00},
    "SelfEmployed": {"NotWorker": 0.00},
    "Employee":     {"NotWorker": 0.00},
}

#H11 #BO
h_employstat_Occupation = {
    "NotWorker":    {"NotWorker": 1.00, "Manager": 0.00, "WhiteC": 0.00, "BlueC": 0.00, "Elementary": 0.00},
    "SelfEmployed": {"NotWorker": 0.00},
    "Employee":     {"NotWorker": 0.00},
}

#H12 #BO
h_employstat_MainTranspWorker = {
    "NotWorker":    {"NotWorker": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "SelfEmployed": {"NotWorker": 0.00},
    "Employee":     {"NotWorker": 0.00},
}

#H13 #BO
h_employstat_TranspTimeWorker = {
    "NotWorker":    {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "SelfEmployed": {"NotWorker": 0.00},
    "Employee":     {"NotWorker": 0.00},
}

# ------------------------------------------------------------------ #
#  STRUCTURAL ZERO-MAPPING: WORKERS (H14 - H20)
# ------------------------------------------------------------------ #
#H14 #BO
h_employment_employstat = {
    "Unemployed": {"NotWorker": 1.00, "SelfEmployed": 0.00, "Employee": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "SelfEmployed": 0.00, "Employee": 0.00},
    "FullTime":   {"NotWorker": 0.00},
    "PartTime":   {"NotWorker": 0.00},
}

#H15 #BO
h_employment_wage = {
    "Unemployed": {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "FullTime":   {"NotWorker": 0.00},
    "PartTime":   {"NotWorker": 0.00},
}

#H16 #BO
h_employment_employcommute = {
    "Unemployed": {"NotWorker": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
    "FullTime":   {"NotWorker": 0.00},
    "PartTime":   {"NotWorker": 0.00},
}

#H17 #BO
h_employment__Profession = {
    "Unemployed": {"NotWorker": 1.00, "AgricFishForest": 0.00, "Industry": 0.00, "Services": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "AgricFishForest": 0.00, "Industry": 0.00, "Services": 0.00},
    "FullTime":   {"NotWorker": 0.00},
    "PartTime":   {"NotWorker": 0.00},
}

#H18 #BO
h_employment__Occupation = {
    "Unemployed": {"NotWorker": 1.00, "Manager": 0.00, "WhiteC": 0.00, "BlueC": 0.00, "Elementary": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "Manager": 0.00, "WhiteC": 0.00, "BlueC": 0.00, "Elementary": 0.00},
    "FullTime":   {"NotWorker": 0.00},
    "PartTime":   {"NotWorker": 0.00},
}

#H19 #BO
h_employment_MainTranspWorker = {
    "Unemployed": {"NotWorker": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "Foot": 0.00, "Bike": 0.00, "PublicTrns": 0.00, "CarDriver": 0.00, "CarPassanger": 0.00, "MotorCycle": 0.00},
    "FullTime":   {"NotWorker": 0.00},
    "PartTime":   {"NotWorker": 0.00},
}

#H20 #BO
h_employment__TranspTimeWorker = {
    "Unemployed": {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "NotInLF":    {"NotWorker": 1.00, "15m-": 0.00, "15-30m": 0.00, "30m+": 0.00},
    "FullTime":   {"NotWorker": 0.00},
    "PartTime":   {"NotWorker": 0.00},
}

#H21 #BO
h_age_BMI = {
    "0-4":    {"UnderAge": 1.00, "UnderWeight": 0.00, "NormalWeight": 0.00, "OverWeight": 0.00, "Obese": 0.00},
    "5-14":   {"UnderAge": 1.00, "UnderWeight": 0.00, "NormalWeight": 0.00, "OverWeight": 0.00, "Obese": 0.00},
    "15-24":  {"UnderAge": 0.00, "UnderWeight": 0.20, "NormalWeight": 0.20, "OverWeight": 0.20, "Obese": 0.40},
    "25-34":  {"UnderAge": 0.00, "UnderWeight": 0.20, "NormalWeight": 0.20, "OverWeight": 0.20, "Obese": 0.40},
    "35-49":  {"UnderAge": 0.00, "UnderWeight": 0.20, "NormalWeight": 0.20, "OverWeight": 0.20, "Obese": 0.40},
    "50-64":  {"UnderAge": 0.00, "UnderWeight": 0.20, "NormalWeight": 0.20, "OverWeight": 0.20, "Obese": 0.40},
    "65-74":  {"UnderAge": 0.00, "UnderWeight": 0.20, "NormalWeight": 0.20, "OverWeight": 0.20, "Obese": 0.40},
    "75+":    {"UnderAge": 0.00, "UnderWeight": 0.20, "NormalWeight": 0.20, "OverWeight": 0.20, "Obese": 0.40},
}



#H22 #BO
h_age_alcoholCons = {
    "0-4":   {"Never": 1.00, "Exceptionally": 0.00, "Consumer": 0.00},
    "5-14":  {"Never": 0.99, "Exceptionally": 0.01, "Consumer": 0.00},
}

#H23 #BO
h_age_smoking = {
    "0-4":   {"Never": 1.00, "Former": 0.00, "1-5": 0.00, "6-10": 0.00, "11-20": 0.00, "20+": 0.00},
    "5-14":  {"Never": 1.00, "Former": 0.00, "1-5": 0.00, "6-10": 0.00, "11-20": 0.00, "20+": 0.00},
}

#H24 #BO
h_SundayOut_age = {
    "Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

#H25 #BO
h_SaturdayOut_age = {
    "Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

#H26 #BO
h_WeekDayOut_age = {
    "Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

#H27 #BO
h_SunSocialEnterT_age = {
    "Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

#H28 #BO
h_SatSocialEnterT_age = {
    "Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

#H29 #BO
h_WeekDSocialEnterT_age = {
    "Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

#H30 #BO
h_SunSportOutD_age = {
    "Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

#H31 #BO
h_SatSportOutD_age = {
    "Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

#H32 #BO
h_WeekDSportOutD_age = {
    "Under3yo": {"0-4": 1.00, "5-14": 0.00, "15-24": 0.00, "25-34": 0.00, "35-49": 0.00, "50-64": 0.00, "65-74": 0.00, "75+": 0.00},
}

#H33 #BO
h_age_LifeSatisfaction = {
    "0-4":  {"Under14yo": 1.00, "0-3": 0.00, "4-6": 0.00, "7-10": 0.00},
    "5-14": {"Under14yo": 1.00, "0-3": 0.00, "4-6": 0.00, "7-10": 0.00},
    "15-24":{"Under14yo": 0.00, "0-3": 0.30, "4-6": 0.30, "7-10": 0.40},
    "25-34":{"Under14yo": 0.00, "0-3": 0.30, "4-6": 0.30, "7-10": 0.40},
    "35-49":{"Under14yo": 0.00, "0-3": 0.30, "4-6": 0.30, "7-10": 0.40},
    "50-64":{"Under14yo": 0.00, "0-3": 0.30, "4-6": 0.30, "7-10": 0.40},
    "65-74":{"Under14yo": 0.00, "0-3": 0.30, "4-6": 0.30, "7-10": 0.40},
    "75+":  {"Under14yo": 0.00, "0-3": 0.30, "4-6": 0.30, "7-10": 0.40},
    
}


#H34 #BO
h_age_employment = {
    "0-4":  {"FullTime": 0.00, "PartTime": 0.00, "Unemployed": 0.00, "NotInLF": 1.00},
    "5-14": {"FullTime": 0.00, "PartTime": 0.00, "Unemployed": 0.00, "NotInLF": 1.00},
}

#H35 #BO
h_age_employstat = {
    "0-4":  {"NotWorker": 1.00, "SelfEmployed": 0.00, "Employee": 0.00},
    "5-14": {"NotWorker": 1.00, "SelfEmployed": 0.00, "Employee": 0.00},
}

#H36 #BO
h_age_Wage = {
    "0-4":  {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
    "5-14": {"NotWorker": 1.00, "Low": 0.00, "Medium": 0.00, "High": 0.00, "VeryHigh": 0.00},
}


# H37 STRUCTURAL TABLES, ROUND 2  --  education <-> student status
h_StudentStat_education = {
    "UniStudent":    {"SecondaryAndLess": 0.0},
    "SchoolStudent": {"Tertiary": 0.0},
}




# H38: P(StudentStat | employment)  #BO 
h_employment_StudentStat = {
    "FullTime": {"NotStudent": 1.00, "SchoolStudent": 0.00, "UniStudent": 0.00},
}

# H39  #BO

h_Occupation_Profession = {
    "Manager":    {"NotWorker": 0.00, "AgricFishForest": 0.04, "Industry": 0.20, "Services": 0.76},
    "WhiteC":     {"NotWorker": 0.00, "AgricFishForest": 0.00, "Industry": 0.25, "Services": 0.75},
    "BlueC":      {"NotWorker": 0.00, "AgricFishForest": 0.04, "Industry": 0.32, "Services": 0.64},
    "Elementary": {"NotWorker": 0.00, "AgricFishForest": 0.24, "Industry": 0.35, "Services": 0.41},
}

#H40 #BO
h_Profession_Occupation = {
    "AgricFishForest": {"NotWorker": 0.00, "Manager": 0.38, "WhiteC": 0.05, "BlueC": 0.55, "Elementary": 0.02},
    "Industry":        {"NotWorker": 0.00, "Manager": 0.16, "WhiteC": 0.31, "BlueC": 0.49, "Elementary": 0.04},
    "Services":        {"NotWorker": 0.00, "Manager": 0.30, "WhiteC": 0.39, "BlueC": 0.25, "Elementary": 0.06},
}



# H41  #BO : commute direction consistency — only the hard geometric impossibilities.
h_employcommute_studentcommute = {
    "Inward":   {"NotStudent": 0.93, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.07},
    "Outward":  {"NotStudent": 1.00, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.00},
}


# H42  #BO : transport identity — off-diagonal zeros when both commuting.
h_MainTranspStudnt_MainTranspWorker = {
    "Foot":        {"Bike": 0.00,
                    "PublicTrns": 0.00, "CarDriver": 0.00,
                    "CarPassanger": 0.00, "MotorCycle": 0.00},
    "Bike":        {"Foot": 0.00,
                    "PublicTrns": 0.00, "CarDriver": 0.00,
                    "CarPassanger": 0.00, "MotorCycle": 0.00},
    "PublicTrns":  {"Foot": 0.00, "Bike": 0.00, "CarDriver": 0.00,
                    "CarPassanger": 0.00, "MotorCycle": 0.00},
    "CarDriver":   {"Foot": 0.00, "Bike": 0.00,
                    "PublicTrns": 0.00,
                    "CarPassanger": 0.00, "MotorCycle": 0.00},
    "CarPassanger":{"Foot": 0.00, "Bike": 0.00,
                    "PublicTrns": 0.00, "CarDriver": 0.00,
                    "MotorCycle": 0.00},
    "MotorCycle":  {"Foot": 0.00, "Bike": 0.00,
                    "PublicTrns": 0.00, "CarDriver": 0.00,
                    "CarPassanger": 0.00},
}



# H43  #BO : a CommuteInward resident cannot have a work commute that is
# internal to the municipality or outward-directed.
h_ResidenceQ_EmployCommute = {
    "CommuteInward": {"NotWorker": 0.55, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.45},
}
 
# H44  #BO : same for the study commute.
h_ResidenceQ_StudentCommute = {
    "CommuteInward": {"NotStudent": 0.75, "InsideBO": 0.00, "Outward": 0.00, "Inward": 0.25},
}
 
# H45  #BO : a CommuteInward resident who is neither a work commuter nor a
# study commuter is impossible. Irreducibly ternary: a non-working student
# commuter and a working non-student commuter are both legitimate, so no
# pairwise rule can express this; only the triple is forbidden.
h_ResidenceQ_EmployCommute_StudentCommute = {
    "CommuteInward": {
        "NotWorker": {"NotStudent": 0.00, "Inward": 1.00},
    },
}


# H46  #BO
h_employstat_StudentStat_LunchPlace = {
    "NotWorker": {"NotStudent": {"Home": 0.33, "Canteen": 0.00, "Restaurant":0.33, "Cafe": 0.34, "AtS/WPlace": 0.00}},
}



# H47 SundayOut <-> SaturdayOut: the Under3yo sentinel must agree  #BO
h_SundayOut_SaturdayOut = {
    "Under3yo": {"ExitHouse": 0.0, "StayIn": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H48 SundayOut <-> WeekDayOut: the Under3yo sentinel must agree  #BO
h_SundayOut_WeekDayOut = {
    "Under3yo": {"ExitHouse": 0.0, "StayIn": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H49 SundayOut <-> SunSocialEnterT: the Under3yo sentinel must agree  #BO
h_SundayOut_SunSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H50 SundayOut <-> SatSocialEnterT: the Under3yo sentinel must agree  #BO
h_SundayOut_SatSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H51 SundayOut <-> WeekDSocialEnterT: the Under3yo sentinel must agree  #BO
h_SundayOut_WeekDSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H52 SundayOut <-> SunSportOutD: the Under3yo sentinel must agree  #BO
h_SundayOut_SunSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H53 SundayOut <-> SatSportOutD: the Under3yo sentinel must agree  #BO
h_SundayOut_SatSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H54 SundayOut <-> WeekDSportOutD: the Under3yo sentinel must agree  #BO
h_SundayOut_WeekDSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H55 SaturdayOut <-> WeekDayOut: the Under3yo sentinel must agree  #BO
h_SaturdayOut_WeekDayOut = {
    "Under3yo": {"ExitHouse": 0.0, "StayIn": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H56 SaturdayOut <-> SunSocialEnterT: the Under3yo sentinel must agree  #BO
h_SaturdayOut_SunSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H57 SaturdayOut <-> SatSocialEnterT: the Under3yo sentinel must agree  #BO
h_SaturdayOut_SatSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H58 SaturdayOut <-> WeekDSocialEnterT: the Under3yo sentinel must agree  #BO
h_SaturdayOut_WeekDSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H59 SaturdayOut <-> SunSportOutD: the Under3yo sentinel must agree  #BO
h_SaturdayOut_SunSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H60 SaturdayOut <-> SatSportOutD: the Under3yo sentinel must agree  #BO
h_SaturdayOut_SatSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H61 SaturdayOut <-> WeekDSportOutD: the Under3yo sentinel must agree  #BO
h_SaturdayOut_WeekDSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H62 WeekDayOut <-> SunSocialEnterT: the Under3yo sentinel must agree  #BO
h_WeekDayOut_SunSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H63 WeekDayOut <-> SatSocialEnterT: the Under3yo sentinel must agree  #BO
h_WeekDayOut_SatSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H64 WeekDayOut <-> WeekDSocialEnterT: the Under3yo sentinel must agree  #BO
h_WeekDayOut_WeekDSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H65 WeekDayOut <-> SunSportOutD: the Under3yo sentinel must agree  #BO
h_WeekDayOut_SunSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H66 WeekDayOut <-> SatSportOutD: the Under3yo sentinel must agree  #BO
h_WeekDayOut_SatSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H67 WeekDayOut <-> WeekDSportOutD: the Under3yo sentinel must agree  #BO
h_WeekDayOut_WeekDSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "ExitHouse": {"Under3yo": 0.0},
    "StayIn": {"Under3yo": 0.0},
}

#H68 SunSocialEnterT <-> SatSocialEnterT: the Under3yo sentinel must agree  #BO
h_SunSocialEnterT_SatSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H69 SunSocialEnterT <-> WeekDSocialEnterT: the Under3yo sentinel must agree  #BO
h_SunSocialEnterT_WeekDSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H70 SunSocialEnterT <-> SunSportOutD: the Under3yo sentinel must agree  #BO
h_SunSocialEnterT_SunSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H71 SunSocialEnterT <-> SatSportOutD: the Under3yo sentinel must agree  #BO
h_SunSocialEnterT_SatSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H72 SunSocialEnterT <-> WeekDSportOutD: the Under3yo sentinel must agree  #BO
h_SunSocialEnterT_WeekDSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H73 SatSocialEnterT <-> WeekDSocialEnterT: the Under3yo sentinel must agree  #BO
h_SatSocialEnterT_WeekDSocialEnterT = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H74 SatSocialEnterT <-> SunSportOutD: the Under3yo sentinel must agree  #BO
h_SatSocialEnterT_SunSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H75 SatSocialEnterT <-> SatSportOutD: the Under3yo sentinel must agree  #BO
h_SatSocialEnterT_SatSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H76 SatSocialEnterT <-> WeekDSportOutD: the Under3yo sentinel must agree  #BO
h_SatSocialEnterT_WeekDSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H77 WeekDSocialEnterT <-> SunSportOutD: the Under3yo sentinel must agree  #BO
h_WeekDSocialEnterT_SunSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H78 WeekDSocialEnterT <-> SatSportOutD: the Under3yo sentinel must agree  #BO
h_WeekDSocialEnterT_SatSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H79 WeekDSocialEnterT <-> WeekDSportOutD: the Under3yo sentinel must agree  #BO
h_WeekDSocialEnterT_WeekDSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H80 SunSportOutD <-> SatSportOutD: the Under3yo sentinel must agree  #BO
h_SunSportOutD_SatSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H81 SunSportOutD <-> WeekDSportOutD: the Under3yo sentinel must agree  #BO
h_SunSportOutD_WeekDSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H82 SatSportOutD <-> WeekDSportOutD: the Under3yo sentinel must agree  #BO
h_SatSportOutD_WeekDSportOutD = {
    "Under3yo": {"Y": 0.0, "N": 0.0, "Under3yo": 1.0},
    "Y": {"Under3yo": 0.0},
    "N": {"Under3yo": 0.0},
}

#H83 Age Studentstat #BO
h_age_StudentStat = {
    "0-4":   {"NotStudent": 1.0, "SchoolStudent": 0.0, "UniStudent": 0.0},
    "5-14":  {"UniStudent": 0.0},
    "25-34": {"SchoolStudent": 0.0},
    "35-49": {"SchoolStudent": 0.0},
    "50-64": {"NotStudent": 1.0, "SchoolStudent": 0.0, "UniStudent": 0.0},
    "65-74": {"NotStudent": 1.0, "SchoolStudent": 0.0, "UniStudent": 0.0},
    "75+":   {"NotStudent": 1.0, "SchoolStudent": 0.0, "UniStudent": 0.0},
}
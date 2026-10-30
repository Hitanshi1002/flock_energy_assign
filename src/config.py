"""
Configuration and Metadata for REFIT Data Cleaning and Appliance Disaggregation.

Contains household metadata, sensor channel mappings, default hyperparameters,
and directory configurations.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOWNLOADS_DIR = PROJECT_ROOT / "data_downloads"
FIGURES_DIR = PROJECT_ROOT / "figures"
CLEANING_FIG_DIR = FIGURES_DIR / "cleaning"
EDA_FIG_DIR = FIGURES_DIR / "eda"
MODEL_FIG_DIR = FIGURES_DIR / "model"

for d in [DATA_DIR, RAW_DATA_DIR, CLEAN_DATA_DIR, PROCESSED_DATA_DIR,
          DOWNLOADS_DIR, FIGURES_DIR, CLEANING_FIG_DIR, EDA_FIG_DIR, MODEL_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset URLs
URL_RAW_7Z = "https://pure.strath.ac.uk/ws/portalfiles/portal/62090177/REFIT_RAW_081116.7z"
URL_CLEAN_7Z = "https://pure.strath.ac.uk/ws/portalfiles/portal/62090184/CLEAN_REFIT_081116.7z"
URL_CLEAN_README = "https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt"
URL_RAW_README = "https://pure.strath.ac.uk/ws/portalfiles/portal/62090176/RAW_READ_ME_081116.txt"

# House metadata from Strathclyde README and MetaData_Tables.xlsx
HOUSE_METADATA = {
    1: {"occupancy": 2, "built": "1975-1980", "type": "Detached", "size": "4 bed", "appliances_count": 35},
    2: {"occupancy": 4, "built": "Unknown", "type": "Semi-detached", "size": "3 bed", "appliances_count": 15},
    3: {"occupancy": 2, "built": "1988", "type": "Detached", "size": "3 bed", "appliances_count": 27},
    4: {"occupancy": 2, "built": "1850-1899", "type": "Detached", "size": "4 bed", "appliances_count": 33},
    5: {"occupancy": 4, "built": "1878", "type": "Mid-terrace", "size": "4 bed", "appliances_count": 44},
    6: {"occupancy": 2, "built": "2005", "type": "Detached", "size": "4 bed", "appliances_count": 49},
    7: {"occupancy": 4, "built": "1965-1974", "type": "Detached", "size": "3 bed", "appliances_count": 25},
    8: {"occupancy": 2, "built": "1966", "type": "Detached", "size": "2 bed", "appliances_count": 35},
    9: {"occupancy": 2, "built": "1919-1944", "type": "Detached", "size": "3 bed", "appliances_count": 24},
    10: {"occupancy": 4, "built": "1919-1944", "type": "Detached", "size": "3 bed", "appliances_count": 31},
    11: {"occupancy": 1, "built": "1945-1964", "type": "Detached", "size": "3 bed", "appliances_count": 25},
    12: {"occupancy": 3, "built": "1991-1995", "type": "Detached", "size": "3 bed", "appliances_count": 26},
    13: {"occupancy": 4, "built": "post 2002", "type": "Detached", "size": "4 bed", "appliances_count": 28},
    15: {"occupancy": 1, "built": "1965-1974", "type": "Semi-detached", "size": "3 bed", "appliances_count": 19},
    16: {"occupancy": 6, "built": "1981-1990", "type": "Detached", "size": "5 bed", "appliances_count": 48},
    17: {"occupancy": 3, "built": "mid 60s", "type": "Detached", "size": "3 bed", "appliances_count": 22},
    18: {"occupancy": 2, "built": "1965-1974", "type": "Detached", "size": "3 bed", "appliances_count": 34},
    19: {"occupancy": 4, "built": "1945-1964", "type": "Semi-detached", "size": "3 bed", "appliances_count": 26},
    20: {"occupancy": 2, "built": "1965-1974", "type": "Detached", "size": "3 bed", "appliances_count": 39},
    21: {"occupancy": 4, "built": "1981-1990", "type": "Detached", "size": "3 bed", "appliances_count": 23},
}

# Appliance Channel Mapping (Appliance 1 to 9)
# Format: {house_id: {channel_number: appliance_name}}
HOUSE_APPLIANCES = {
    1: {
        1: "Fridge",
        2: "Freezer_1",
        3: "Freezer_2",
        4: "Washer_Dryer",
        5: "Washing_Machine",
        6: "Dishwasher",
        7: "Computer",
        8: "Television",
        9: "Electric_Heater"
    },
    2: {
        1: "Fridge_Freezer",
        2: "Washing_Machine",
        3: "Dishwasher",
        4: "Television",
        5: "Microwave",
        6: "Toaster",
        7: "Hi_Fi",
        8: "Kettle",
        9: "Overhead_Fan"
    },
    3: {
        1: "Toaster",
        2: "Fridge_Freezer",
        3: "Freezer",
        4: "Tumble_Dryer",
        5: "Dishwasher",
        6: "Washing_Machine",
        7: "Television",
        8: "Microwave",
        9: "Kettle"
    },
    5: {
        1: "Fridge_Freezer",
        2: "Tumble_Dryer",
        3: "Washing_Machine",
        4: "Dishwasher",
        5: "Desktop_Computer",
        6: "Television",
        7: "Microwave",
        8: "Kettle",
        9: "Toaster"
    },
    8: {
        1: "Fridge",
        2: "Freezer",
        3: "Washer_Dryer",
        4: "Washing_Machine",
        5: "Toaster",
        6: "Computer",
        7: "Television",
        8: "Microwave",
        9: "Kettle"
    },
    9: {
        1: "Fridge_Freezer",
        2: "Washer_Dryer",
        3: "Washing_Machine",
        4: "Dishwasher",
        5: "Television",
        6: "Microwave",
        7: "Kettle",
        8: "Hi_Fi",
        9: "Electric_Heater"
    },
    20: {
        1: "Fridge",
        2: "Freezer",
        3: "Tumble_Dryer",
        4: "Washing_Machine",
        5: "Dishwasher",
        6: "Computer",
        7: "Television",
        8: "Microwave",
        9: "Kettle"
    }
}

# Washing Machine Channel Index for relevant households
WASHING_MACHINE_CHANNELS = {
    1: 5,
    2: 2,
    3: 6,
    5: 3,
    8: 4,
    9: 3,
    10: 5,
    15: 3,
    16: 5,
    17: 4,
    18: 5,
    19: 2,
    20: 4,
    21: 3,
}

# Primary target households for analysis
SELECTED_EDA_HOUSES = [1, 2, 3, 20]

# Cleaning Pipeline Hyperparameters
MAX_VALID_AGGREGATE_POWER = 15000.0  # Watts
MAX_VALID_IAM_POWER = 4000.0        # Watts
MAX_INTERPOLATION_MINUTES = 3       # Short gaps filled up to 3 minutes
OUTAGE_THRESHOLD_MINUTES = 15       # Gaps > 15 mins flagged as severe outage

# Cycle Detection Hyperparameters
CYCLE_ACTIVE_POWER_THRESHOLD = 15.0  # W (standby vs active)
CYCLE_HEATING_PEAK_THRESHOLD = 500.0 # W (heating element spike confirmation)
CYCLE_MAX_PAUSE_MINUTES = 3          # Max duration below threshold during soaking/pause
CYCLE_MIN_DURATION_MINUTES = 15      # Minimum active cycle duration
CYCLE_MIN_ENERGY_KWH = 0.05          # Minimum cycle energy consumption (filter false triggers)

# Model Features & Windowing
ROLLING_WINDOWS_MINUTES = [5, 15, 30, 60]
LAGS_LEADS_MINUTES = [-15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15]

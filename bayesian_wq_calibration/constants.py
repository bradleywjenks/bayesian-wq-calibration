from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data/'
NETWORK_DIR = DATA_DIR / 'network/'
RESULTS_DIR = BASE_DIR / 'results/'
DEVICE_DIR = DATA_DIR / 'devices/'
TIMESERIES_DIR = DATA_DIR / 'timeseries/'
INP_FILE = 'BWFLnet_2023_04.inp'
SPLIT_INP_FILE = 'BWFLnet_2023_04_split_and_ivs_removed.inp'

IV_CLOSE = 1e7
IV_OPEN = 1e-4
IV_OPEN_PART = 4e2


# print paths for verification
if __name__ == "__main__":
    print("Base Directory:", BASE_DIR)
    print("Raw Data Directory:", DATA_DIR)
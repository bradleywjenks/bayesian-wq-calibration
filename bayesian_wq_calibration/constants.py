from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data/'
NETWORK_DIR = DATA_DIR / 'network/'
DEVICE_DIR = DATA_DIR / 'devices/'
TIMESERIES_DIR = DATA_DIR / 'timeseries/'


# print paths for verification
if __name__ == "__main__":
    print("Base Directory:", BASE_DIR)
    print("Raw Data Directory:", DATA_DIR)
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
ROLLING_WINDOWS_DATA_DIR = DATA_DIR / "rolling_windows"
ROLLING_WINDOWS_PLOTS_DIR = PLOTS_DIR / "rolling_windows"
WORKBOOK = DATA_DIR / "Backtest-Portfolio-returns-rev25c.xlsx"


@dataclass(frozen=True)
class DatasetVariant:
    key: str
    start_year: int
    label: str
    title_suffix: str

    @property
    def data_dir(self) -> Path:
        return ROLLING_WINDOWS_DATA_DIR / self.key

    @property
    def plots_dir(self) -> Path:
        return ROLLING_WINDOWS_PLOTS_DIR / self.key


DATASET_VARIANTS = {
    "full_history": DatasetVariant(
        key="full_history",
        start_year=1871,
        label="Full History",
        title_suffix="Full History",
    ),
    "from_1927": DatasetVariant(
        key="from_1927",
        start_year=1927,
        label="1927 Onward",
        title_suffix="Since 1927",
    ),
}


def get_dataset_variant(dataset: str) -> DatasetVariant:
    try:
        return DATASET_VARIANTS[dataset]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_VARIANTS))
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {available}") from exc

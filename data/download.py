from pathlib import Path

import pandas as pd

SOURCE = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-02-11/hotels.csv"
)


def main() -> None:
    output_dir = Path(__file__).parent
    frame = pd.read_csv(SOURCE)
    frame.to_csv(output_dir / "hotel_bookings.csv", index=False)


if __name__ == "__main__":
    main()

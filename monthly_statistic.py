import polars as pl
from pathlib import Path
from typing import List

assert pl.__version__ == "0.18.0"

month_dfs: List[pl.DataFrame] = []
for f in Path("町洋產品/").iterdir():
    if not f.is_file() or ".csv" not in f.name:
        continue
    print(f)
    df = pl.read_csv(f, dtypes={"Object": pl.Utf8})
    df = (
        df.lazy()
        .select(["month", "Qty", "Amount", "ProductCode", "EntityCode"])
        .filter(pl.col("Qty") != 0)
        .with_columns((pl.col("Amount") / pl.col("Qty")).alias("unit cost"))
        .select(["month", "unit cost", "ProductCode", "EntityCode"])
        .collect()
    )
    month_dfs.append(df)

df = pl.concat(month_dfs)
# print(df)
# shape: (9_725_970, 4)
# ┌───────┬────────────┬───────────────┬──────────────┐
# │ month ┆ unit cost  ┆ ProductCode   ┆ EntityCode   │
# │ ---   ┆ ---        ┆ ---           ┆ ---          │
# │ i64   ┆ f64        ┆ str           ┆ str          │
# ╞═══════╪════════════╪═══════════════╪══════════════╡
# │ 12    ┆ 0.000047   ┆ 0001B00700    ┆ A05150500323 │
# │ 12    ┆ 0.000023   ┆ 0001B00700    ┆ A20040400323 │
# │ 12    ┆ 0.000008   ┆ 0001B00700    ┆ A20200460323 │
# │ 12    ┆ 0.000137   ┆ 0001B00700    ┆ A30100350323 │
# │ …     ┆ …          ┆ …             ┆ …            │
# │ 11    ┆ 23.155703  ┆ X012-A63RV0-1 ┆ 5120000302   │
# │ 11    ┆ 0.222181   ┆ X013-A30SF-1  ┆ A60107000303 │
# │ 11    ┆ 20.283266  ┆ X013-A30SF-1  ┆ A60200600303 │
# │ 11    ┆ 211.312645 ┆ X013-A30SF-1  ┆ 0000-A30SF-0 │
# └───────┴────────────┴───────────────┴──────────────┘


df = (
    df.lazy()
    .groupby(["month", "ProductCode", "EntityCode"])  # match this unique patter
    .agg(pl.col("unit cost").mean())
    .sort(["ProductCode", "EntityCode", "month"])
    .groupby(["ProductCode", "EntityCode"], maintain_order=True)
    .agg([pl.col("month"), pl.col("unit cost")])  # agg 2 col to a list
    .with_columns(
        # create boolean mask list of each month
        [
            pl.col("month")
            .list.eval((pl.element() == m), parallel=True)
            .alias(f"{m}_idx")
            for m in range(1, 13)
        ]
        # ┌───────┬─────────────┬───────┬────────────────────────┐
        # │       ┆ month       ┆       ┆ 1_idx                  │
        # │ ..... ┆ ---         ┆ ..... ┆ ---                    │
        # │       ┆ list[i64]   ┆       ┆ list[bool]             │
        # ╞═══════╪═════════════╪═══════╪════════════════════════╡
        # │       ┆ [5]         ┆       ┆ [false]                │
        # │       ┆ [1, 2, … 9] ┆       ┆ [true, false, … false] │
        # │       ┆ [1, 2, … 9] ┆       ┆ [true, false, … false] │
        # │       ┆ [1, 5]      ┆       ┆ [true, false]          │
        # │ ..... ┆ …           ┆ ..... ┆ …                      │
        # │       ┆ [9]         ┆       ┆ [false]                │
        # │       ┆ [9]         ┆       ┆ [false]                │
        # │       ┆ [9]         ┆       ┆ [false]                │
        # │       ┆ [9]         ┆       ┆ [false]                │
        # └───────┴─────────────┴───────┴────────────────────────┘
    )
    # explode list to each row
    .explode(["unit cost"] + [f"{m}_idx" for m in range(1, 13)])
    .groupby(["ProductCode", "EntityCode"], maintain_order=True)
    # agg back to unique ProductCode + Entity
    .agg(
        [
            # agg get month unit cost by index
            pl.col("unit cost").filter(pl.col(f"{m}_idx")).alias(f"{m}")
            for m in range(1, 13)
        ]
    )
    # explode single value list to value only
    .with_columns([pl.col(f"{m}").explode() for m in range(1, 13)])
    .collect()
)
# ┌──────────────┬──────────────┬──────────┬──────────┬───┬──────┬──────┐
# │ ProductCode  ┆ EntityCode   ┆ 1        ┆ 2        ┆ … ┆ 11   ┆ 12   │
# │ ---          ┆ ---          ┆ ---      ┆ ---      ┆   ┆ ---  ┆ ---  │
# │ str          ┆ str          ┆ f64      ┆ f64      ┆   ┆ f64  ┆ f64  │
# ╞══════════════╪══════════════╪══════════╪══════════╪═══╪══════╪══════╡
# │ 0001B00400   ┆ 0001B00400   ┆ null     ┆ null     ┆ … ┆ null ┆ null │
# │ 0001B00400   ┆ 0001B00400-S ┆ 0.566976 ┆ 0.541101 ┆ … ┆ null ┆ null │
# │ 0001B00400   ┆ 5120000104C  ┆ 0.012317 ┆ 0.019499 ┆ … ┆ null ┆ null │
# │ 0001B00400   ┆ 5120000302   ┆ 0.102842 ┆ null     ┆ … ┆ null ┆ null │
# │ …            ┆ …            ┆ …        ┆ …        ┆ … ┆ …    ┆ …    │
# │ X013-E471I-1 ┆ A99150800093 ┆ null     ┆ null     ┆ … ┆ null ┆ null │
# │ X013-E471I-1 ┆ SF-LS107-N   ┆ null     ┆ null     ┆ … ┆ null ┆ null │
# │ X013-E471I-1 ┆ SF-LSG54-X   ┆ null     ┆ null     ┆ … ┆ null ┆ null │
# │ X013-E471I-1 ┆ SF-W101-W    ┆ null     ┆ null     ┆ … ┆ null ┆ null │
# └──────────────┴──────────────┴──────────┴──────────┴───┴──────┴──────┘

df = pl.concat(
    [
        df,
        df.select([f"{m}" for m in range(1, 4)]).mean(axis=1).to_frame("Q1"),
        df.select([f"{m}" for m in range(4, 7)]).mean(axis=1).to_frame("Q2"),
        df.select([f"{m}" for m in range(7, 10)]).mean(axis=1).to_frame("Q3"),
        df.select([f"{m}" for m in range(10, 13)]).mean(axis=1).to_frame("Q4"),
        df.select([f"{m}" for m in range(1, 7)])
        .mean(axis=1)
        .to_frame("first half year"),
        df.select([f"{m}" for m in range(8, 13)])
        .mean(axis=1)
        .to_frame("second half year"),
        df.select([f"{m}" for m in range(1, 13)])
        .mean(axis=1)
        .to_frame("one year"),
    ],
    how="horizontal",
)

print(df)
df.drop([f"{m}" for m in range(1, 13)]).write_csv("p.csv")

import pandas as pd

# Load production dataset
prod = pd.read_csv("maine_crop_production_county_2007_2012_2017_2022.csv")

#Load weather dataset
weather = pd.read_csv("maine_allstations_temp_precip_2007_2012_2017_2022_year.csv")

# Group by year + county and compute mean production
prod_grouped = (
    prod.groupby(["year", "county_name"], as_index=False)["Value_num"]
        .mean()
        .rename(columns={"Value_num": "avg_production"})
)
# Save as a NEW dataset
prod_grouped.to_csv("maine_avg_production_by_county_year.csv", index=False)

print("Saved new dataset: maine_avg_production_by_county_year.csv")
print(prod_grouped.head())

prod_grouped_total = (
    prod.groupby(["year", "county_name"], as_index=False)["Value_num"]
        .sum()
        .rename(columns={"Value_num": "total_production"})
)

# Save as a NEW dataset
prod_grouped_total.to_csv("maine_total_production_by_county_year.csv", index=False)

print("Saved new dataset: maine_total_production_by_county_year.csv")
print(prod_grouped_total.head())




weather_grouped = (
    weather.groupby(["year", "county"], as_index=False)
    .agg({
        "temp_lo_F": "mean",
        "temp_hi_F": "mean",
        "temp_avg_F": "mean",
        "precip_1wk_inches": "sum",
        "precip_1wk_days": "sum"
    })
)

weather_grouped = weather_grouped.rename(columns={
    "precip_1wk_inches": "total_precip_inches",
    "precip_1wk_days": "total_precip_days"
})

# Save aggregated file
weather_grouped.to_csv("maine_weather_aggregated_by_county_year.csv", index=False)

print("Created weather file: maine_weather_aggregated_by_county_year.csv")
print(weather_grouped.head())


# Merge directly using matching names
merged = prod_grouped_total.merge(
    weather_grouped,
    left_on=["year", "county_name"],   # production columns
    right_on=["year", "county"],       # weather columns
    how="inner"
)

# More cleanup
merged = merged.drop(columns=["county"])  # remove duplicate
merged = merged.rename(columns={"county_name": "county"})

# Save the final merged dataset
merged.to_csv("maine_merged_total_production_weather.csv", index=False)

print("Saved merged dataset: maine_merged_total_production_weather.csv")
print(merged.head())


import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from dataloader import MIMICIVDataLoader


class SepsisRLDataset:

    def __init__(self, loader: MIMICIVDataLoader):
        self.loader = loader
        self.cohort = None
        self.timeseries = None
        self.state_matrix = None
        self.cluster_labels = None

    def build_sepsis_cohort(self, nrows=None):

        icustays = self.loader.load_table("icustays", nrows=nrows)
        admissions = self.loader.load_table("admissions", nrows=nrows)
        diagnoses = self.loader.load_table("diagnoses_icd", nrows=nrows)

        # Sepsis ICD10
        sepsis_codes = ["A40", "A41"]

        diagnoses["is_sepsis"] = diagnoses["icd_code"].str.startswith(tuple(sepsis_codes))

        sepsis_hadm = diagnoses.loc[
            diagnoses["is_sepsis"], "hadm_id"
        ].unique()

        cohort = icustays[icustays["hadm_id"].isin(sepsis_hadm)]

        self.cohort = cohort
        print("Sepsis ICU stays:", cohort.shape)

        return cohort
    
    def extract_vitals(self, nrows=None):

        d_items = self.loader.load_table("d_items", nrows=nrows)
        chartevents = self.loader.load_table("chartevents", nrows=nrows)

        vital_labels = [
            "Heart Rate",
            "Respiratory Rate",
            "O2 saturation pulseoxymetry",
            "Non Invasive Blood Pressure systolic",
            "Non Invasive Blood Pressure diastolic",
            "Temperature Fahrenheit"
        ]

        vital_items = d_items[d_items["label"].isin(vital_labels)]

        chartevents = chartevents[
            chartevents["itemid"].isin(vital_items["itemid"])
        ]

        chartevents = chartevents[
            chartevents["stay_id"].isin(self.cohort["stay_id"])
        ]

        icu_times = self.cohort[["stay_id", "intime"]].copy()
        icu_times["intime"] = pd.to_datetime(icu_times["intime"])

        chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])

        chartevents = chartevents.merge(
            icu_times,
            on="stay_id",
            how="left"
        )

        chartevents["hours_since_icu"] = (
            chartevents["charttime"] - chartevents["intime"]
        ).dt.total_seconds() / 3600

        self.timeseries = chartevents

        return chartevents
    
    def discretize_4h(self):

        df = self.timeseries.copy()

        df = df[df["hours_since_icu"] >= 0]

        df["block"] = (df["hours_since_icu"] // 4).astype(int)

        # first 48h
        df = df[df["block"] <= 12]

        self.timeseries = df

        return df

    def aggregate_blocks(self):

        df = self.timeseries.copy()

        if "stay_id" not in df.columns:
            raise ValueError("stay_id column missing — check extract_vitals()")

        agg = df.groupby(
            ["stay_id", "block", "itemid"]
        )["valuenum"].mean().reset_index()

        pivot = agg.pivot_table(
            index=["stay_id", "block"],
            columns="itemid",
            values="valuenum"
        )

        pivot = pivot.reset_index()

        self.state_matrix = pivot

        return pivot

    def cluster_states(self, n_clusters=50):

        X = self.state_matrix.drop(columns=["stay_id", "block"])

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=42
        )

        labels = kmeans.fit_predict(X_scaled)

        self.state_matrix["state"] = labels
        self.cluster_labels = labels

        print("States created:", len(np.unique(labels)))

        return self.state_matrix
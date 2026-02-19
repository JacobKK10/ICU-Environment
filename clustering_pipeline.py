import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from dataloader import MIMICIVDataLoader


class ICUClusteringPipeline:

    def __init__(self, loader: MIMICIVDataLoader):
        self.loader = loader
        self.data = None
        self.X = None
        self.X_scaled = None
        self.clusters = None

    def build_basic_cohort(self, nrows=None):

        icustays = self.loader.load_table(
            "icustays",
            nrows=nrows
        )

        admissions = self.loader.load_table(
            "admissions",
            nrows=nrows
        )

        patients = self.loader.load_table(
            "patients",
            nrows=nrows
        )

        df = icustays.merge(admissions, on=["subject_id", "hadm_id"])
        df = df.merge(patients, on="subject_id")

        df["los_icu_days"] = (
            pd.to_datetime(df["outtime"]) -
            pd.to_datetime(df["intime"])
        ).dt.total_seconds() / (3600 * 24)

        df["mortality"] = df["hospital_expire_flag"]

        self.data = df
        return df

    def select_features(self):

        features = [
            "anchor_age",
            "los_icu_days",
            "mortality"
        ]

        df = self.data[features].copy()

        df["gender"] = pd.get_dummies(
            self.data["gender"],
            drop_first=True
        )

        self.X = df
        return df

    def preprocess(self):

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(self.X)

        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X_imputed)

        return self.X_scaled

    def run_kmeans(self, n_clusters=4):

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42
        )

        self.clusters = kmeans.fit_predict(self.X_scaled)
        self.data["cluster"] = self.clusters

        score = silhouette_score(self.X_scaled, self.clusters)

        print("Silhouette score:", score)

        return self.data

    def cluster_summary(self):

        numeric_cols = self.data.select_dtypes(include=np.number).columns

        summary = self.data.groupby("cluster")[numeric_cols].mean()

        print(summary)

        return summary
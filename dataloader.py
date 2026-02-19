import os
import pandas as pd
import numpy as np
from typing import Optional, Dict


class MIMICIVDataLoader:

    def __init__(self, base_path: str, load_hosp: bool = True):
        self.base_path = base_path
        self.icu_path = os.path.join(base_path, "icu")
        self.hosp_path = os.path.join(base_path, "hosp")

        self.load_hosp = load_hosp
        self.tables: Dict[str, pd.DataFrame] = {}

        self.icu_files = {
            "caregiver": "caregiver.csv.gz",
            "chartevents": "chartevents.csv.gz",
            "d_items": "d_items.csv.gz",
            "datetimeevents": "datetimeevents.csv.gz",
            "icustays": "icustays.csv.gz",
            "ingredientevents": "ingredientevents.csv.gz",
            "inputevents": "inputevents.csv.gz",
            "outputevents": "outputevents.csv.gz",
            "procedureevents": "procedureevents.csv.gz"
        }

        self.hosp_files = {
            "admissions": "admissions.csv.gz",
            "d_hcpcs": "d_hcpcs.csv.gz",
            "d_icd_diagnoses": "d_icd_diagnoses.csv.gz",
            "d_icd_procedures": "d_icd_procedures.csv.gz",
            "d_labitems": "d_labitems.csv.gz",
            "diagnoses_icd": "diagnoses_icd.csv.gz",
            "drgcodes": "drgcodes.csv.gz",
            "emar_detail": "emar_detail.csv.gz",
            "emar": "emar.csv.gz",
            "hcpcsevents": "hcpcsevents.csv.gz",
            "labevents": "labevents.csv.gz",
            "microbiologyevents": "microbiologyevents.csv.gz",
            "omr": "omr.csv.gz",
            "patients": "patients.csv.gz",
            "pharmacy": "pharmacy.csv.gz",
            "poe_detail": "poe_detail.csv.gz",
            "poe": "poe.csv.gz",
            "prescriptions": "prescriptions.csv.gz",
            "procedures_icd": "procedures_icd.csv.gz",
            "provider": "provider.csv.gz",
            "services": "services.csv.gz",
            "transfers": "transfers.csv.gz"
        }

    def load_table(
        self,
        table_name: str,
        nrows: Optional[int] = None,
        usecols: Optional[list] = None
    ) -> pd.DataFrame:

        # ICU
        if table_name in self.icu_files:
            path = os.path.join(self.icu_path, self.icu_files[table_name])

        # HOSP
        elif table_name in self.hosp_files and self.load_hosp:
            path = os.path.join(self.hosp_path, self.hosp_files[table_name])

        else:
            raise ValueError(f"Table '{table_name}' doesn't exist.")

        print(f"Loading table {table_name}")

        df = pd.read_csv(
            path,
            compression="gzip",
            nrows=nrows,
            usecols=usecols
        )

        self.tables[table_name] = df
        return df

    def get(self, table_name: str) -> pd.DataFrame:
        if table_name not in self.tables:
            return self.load_table(table_name)

        return self.tables[table_name]

    def load_all(self, nrows: Optional[int] = None):
        for table in self.icu_files:
            self.load_table(table, nrows=nrows)

        if self.load_hosp:
            for table in self.hosp_files:
                self.load_table(table, nrows=nrows)

        return self.tables

    def clear(self):
        self.tables = {}
        print("All loaded data cleared from memory.")

    
    def eda_report(self, table_name: str) -> pd.DataFrame:

        df = self.get(table_name)

        print(f"\nEDA: {table_name}")
        print("Shape:", df.shape)

        print("\nColumn types:")
        print(df.dtypes)

        print("\nMissing values summary:")
        missing = df.isna().sum()
        missing_percent = (missing / len(df)) * 100

        report = pd.DataFrame({
            "missing_count": missing,
            "missing_percent": missing_percent
        }).sort_values("missing_percent", ascending=False)

        print(report.head(20))

        print("\nDuplicated rows:", df.duplicated().sum())

        print("\nBasic statistics (numerical columns):")
        print(df.describe().T)

        return report
    
    def missing_analysis(self, table_name: str, threshold: float = 50.0):
        df = self.get(table_name)

        missing_percent = (df.isna().sum() / len(df)) * 100
        high_missing = missing_percent[missing_percent > threshold]

        print(f"\nColumns with > {threshold}% missing:")
        print(high_missing.sort_values(ascending=False))

    def numeric_distribution(self, table_name: str):
        df = self.get(table_name)

        numeric_cols = df.select_dtypes(include=np.number).columns

        print("\nNumeric column distribution summary:")
        print(df[numeric_cols].describe().T)

    def categorical_summary(self, table_name: str, top_n: int = 10):
        df = self.get(table_name)

        cat_cols = df.select_dtypes(include=["object"]).columns

        for col in cat_cols:
            print(f"\nColumn: {col}")
            print(df[col].value_counts().head(top_n))



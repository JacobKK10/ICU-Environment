# from dataloader import MIMICIVDataLoader
# from clustering_pipeline import ICUClusteringPipeline

# path = "D:\\Studia D\\Praca magisterska D\\mimic-iv-3.1"

# loader = MIMICIVDataLoader(path, load_hosp=True)
# pipeline = ICUClusteringPipeline(loader)

# loader.load_table("icustays", nrows = 50000)
# loader.eda_report("icustays")
# loader.missing_analysis("icustays")
# loader.numeric_distribution("icustays")
# loader.categorical_summary("icustays")

# pipeline.build_basic_cohort(nrows=50000)
# pipeline.select_features()
# pipeline.preprocess()
# clustered = pipeline.run_kmeans(n_clusters=4)

# pipeline.cluster_summary()

# loader.clear()

from dataloader import MIMICIVDataLoader
from sepsis_rl_dataset import SepsisRLDataset

path = "D:\\Studia D\\Praca magisterska D\\mimic-iv-3.1"

loader = MIMICIVDataLoader(path, load_hosp=True)

builder = SepsisRLDataset(loader)

builder.build_sepsis_cohort()
builder.extract_vitals(nrows=2_000_000)
builder.discretize_4h()
builder.aggregate_blocks()
states = builder.cluster_states(n_clusters=50)

print(states.head())

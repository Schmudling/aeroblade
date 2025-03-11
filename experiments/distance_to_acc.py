import pandas as pd
from sklearn.metrics import average_precision_score
from pathlib import Path

# Define the input and output paths
input_file = "distances.csv"
output_dir = Path("output")  # Change this to your desired output directory
output_dir.mkdir(exist_ok=True)

# Read CSV file
distances = pd.read_csv(input_file)

# Ensure "distance" column is numeric
distances["distance"] = pd.to_numeric(distances["distance"], errors="coerce")


categoricals = [
        "dir",
        "image_size",
        "repo_id",
        "transform",
        "distance_metric",
        "file",
    ]
    distances[categoricals] = distances[categoricals].astype("category")
    distances.to_parquet(output_dir / "distances.parquet")

    # compute detection results
    detection_results = []
    for (transform, repo_id, dist_metric), group_df in distances.groupby(
        ["transform", "repo_id", "distance_metric"], sort=False, observed=True
    ):
        y_score_real = group_df.query("dir == @args.real_dir.__str__()").distance.values
        for fake_dir in args.fake_dirs:
            y_score_fake = group_df.query("dir == @fake_dir.__str__()").distance.values
            y_score = y_score_real.tolist() + y_score_fake.tolist()
            y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)
            ap = average_precision_score(y_true=y_true, y_score=y_score)
            tpr5fpr = tpr_at_max_fpr(y_true=y_true, y_score=y_score, max_fpr=0.05)
            detection_results.append(
                {
                    "fake_dir": fake_dir,
                    "transform": transform,
                    "repo_id": repo_id,
                    "distance_metric": dist_metric,
                    "ap": ap,
                    "tpr5fpr": tpr5fpr,
                }
            )
    pd.DataFrame(detection_results).sort_values("fake_dir", kind="stable").to_csv(
        output_dir / "detection_results.csv"
    )

    # compute attribution results
    attribution_results = []
    for (dir, transform, dist_metric), group_df in distances.groupby(
        ["dir", "transform", "distance_metric"], sort=False, observed=True
    ):
        for repo_id, repo_id_df in group_df.groupby(
            "repo_id", sort=False, observed=True
        ):
            if repo_id == "max":
                continue
            matches = (
                repo_id_df.distance.values
                == group_df.query("repo_id == 'max'").distance.values
            )
            fraction = matches.sum() / len(repo_id_df)
            attribution_results.append(
                {
                    "dir": dir,
                    "transform": transform,
                    "distance_metric": dist_metric,
                    "repo_id": repo_id,
                    "fraction": fraction,
                }
            )
    pd.DataFrame(attribution_results).sort_values("dir", kind="stable").to_csv(
        output_dir / "attribution_results.csv"
    )

    print("Done!")

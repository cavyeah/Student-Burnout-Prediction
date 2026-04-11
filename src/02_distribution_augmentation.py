"""Distribution-based augmentation pipeline for student burnout data.

This script performs controlled synthetic augmentation before CTGAN by anchoring
its sampling to real behavioral distributions and class-wise relationships.
It uses only NumPy, Pandas, and standard-library modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 42
TARGET_CLASS_SIZE = 40
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "burnout_balanced_preCTGAN.csv"

REQUIRED_MAIN_COLUMNS = [
    "EE",
    "DP",
    "PA",
    "Average sleep hours",
    "Screen time",
    "Daily study hours",
    "Procastination",
    "Burnout",
]

REQUIRED_DATASET_B_COLUMNS = [
    "Sleep Duration",
    "Screen Time",
    "Physical Activity",
    "Stress Level",
]

NUMERIC_MAIN_COLUMNS = [
    "EE",
    "DP",
    "PA",
    "Average sleep hours",
    "Screen time",
    "Daily study hours",
    "Procastination",
]

LOGGER = logging.getLogger("distribution_augmentation")
AUGMENTATION_CONTEXT: "AugmentationContext | None" = None


@dataclass
class AugmentationContext:
    """Container for the statistics used during synthetic generation."""

    rng: np.random.Generator
    main_df: pd.DataFrame
    main_profiles: dict[int, dict[str, dict[str, float]]]
    global_main_profile: dict[str, dict[str, float]]
    dataset_b_profile: dict[str, float]
    stress_probabilities: np.ndarray
    output_columns: list[str]
    class_counts: pd.Series
    class_order: list[int]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed main, Dataset A, and Dataset B files.

    Raises:
        FileNotFoundError: If any required input file is missing.
        ValueError: If required columns are not present.

    Returns:
        A tuple of (main_df, dataset_a_df, dataset_b_df).
    """
    main_path = PROCESSED_DIR / "burnout_cleaned.csv"
    dataset_a_path = PROCESSED_DIR / "dataset_A_cleaned.csv"
    dataset_b_path = PROCESSED_DIR / "dataset_B_cleaned.csv"

    missing_files = [path for path in [main_path, dataset_a_path, dataset_b_path] if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing processed input file(s): "
            + ", ".join(str(path) for path in missing_files)
        )

    main_df = pd.read_csv(main_path)
    dataset_a_df = pd.read_csv(dataset_a_path)
    dataset_b_df = pd.read_csv(dataset_b_path)

    _validate_columns(main_df, REQUIRED_MAIN_COLUMNS, "main dataset")
    _validate_columns(dataset_b_df, REQUIRED_DATASET_B_COLUMNS, "Dataset B")

    LOGGER.info("Loaded main dataset: %s", main_df.shape)
    LOGGER.info("Loaded Dataset A: %s", dataset_a_df.shape)
    LOGGER.info("Loaded Dataset B: %s", dataset_b_df.shape)

    return main_df, dataset_a_df, dataset_b_df


def _validate_columns(frame: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    """Ensure a dataframe contains the required columns."""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required column(s): {', '.join(missing)}"
        )


def extract_distributions(
    main_df: pd.DataFrame,
    dataset_a_df: pd.DataFrame,
    dataset_b_df: pd.DataFrame,
) -> AugmentationContext:
    """Extract the distributions that drive augmentation.

    Dataset B provides the anchor distributions for sleep, screen time, physical
    activity, and stress. The main dataset provides class-wise profiles for the
    output schema so the generated rows preserve realistic relationships.

    Returns:
        A fully populated AugmentationContext instance.
    """
    rng = np.random.default_rng(SEED)

    dataset_b_profile = {
        "sleep_mean": float(dataset_b_df["Sleep Duration"].mean()),
        "sleep_std": float(dataset_b_df["Sleep Duration"].std(ddof=0) or 1.0),
        "screen_mean": float(dataset_b_df["Screen Time"].mean()),
        "screen_std": float(dataset_b_df["Screen Time"].std(ddof=0) or 1.0),
        "activity_mean": float(dataset_b_df["Physical Activity"].mean()),
        "activity_std": float(dataset_b_df["Physical Activity"].std(ddof=0) or 1.0),
    }

    stress_counts = (
        dataset_b_df["Stress Level"]
        .value_counts(normalize=True)
        .reindex([0, 1, 2], fill_value=0.0)
        .astype(float)
    )
    stress_probabilities = stress_counts.to_numpy(dtype=float)
    if np.isclose(stress_probabilities.sum(), 0.0):
        stress_probabilities = np.array([0.2, 0.6, 0.2], dtype=float)
    else:
        stress_probabilities = stress_probabilities / stress_probabilities.sum()

    class_order = sorted(int(value) for value in main_df["Burnout"].dropna().unique())
    if not class_order:
        raise ValueError("Main dataset does not contain any burnout classes.")

    global_main_profile = _build_profile(main_df)
    main_profiles: dict[int, dict[str, dict[str, float]]] = {}
    for class_label in class_order:
        class_frame = main_df[main_df["Burnout"] == class_label]
        if class_frame.empty:
            class_frame = main_df
        main_profiles[class_label] = _build_profile(class_frame)

    LOGGER.info("Dataset B anchors: %s", dataset_b_profile)
    LOGGER.info("Stress probabilities from Dataset B: %s", stress_probabilities.tolist())
    LOGGER.info("Detected burnout classes: %s", class_order)

    if set(dataset_a_df.columns).intersection({"social_support", "academic_performance"}):
        LOGGER.info("Dataset A loaded successfully and available for extension if needed.")

    return AugmentationContext(
        rng=rng,
        main_df=main_df,
        main_profiles=main_profiles,
        global_main_profile=global_main_profile,
        dataset_b_profile=dataset_b_profile,
        stress_probabilities=stress_probabilities,
        output_columns=list(main_df.columns),
        class_counts=main_df["Burnout"].value_counts().sort_index(),
        class_order=class_order,
    )


def _build_profile(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build mean/std profiles for numeric columns in a dataframe."""
    profile: dict[str, dict[str, float]] = {}
    for column in NUMERIC_MAIN_COLUMNS:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            mean_value = 0.0
            std_value = 1.0
        else:
            mean_value = float(series.mean())
            std_value = float(series.std(ddof=0))
            if not np.isfinite(std_value) or np.isclose(std_value, 0.0):
                std_value = max(abs(mean_value) * 0.05, 0.5)
        profile[column] = {"mean": mean_value, "std": std_value}
    return profile


def generate_samples(n: int, class_label: int) -> pd.DataFrame:
    """Generate synthetic rows for a given burnout class.

    The function uses class-specific base profiles, Gaussian sampling, and a
    shared latent factor so the generated rows retain meaningful correlations.

    Args:
        n: Number of rows to generate.
        class_label: Burnout class label (0, 1, or 2).

    Returns:
        A dataframe containing n synthetic rows.
    """
    if AUGMENTATION_CONTEXT is None:
        raise RuntimeError("Augmentation context has not been initialized.")

    if n <= 0:
        return pd.DataFrame(columns=AUGMENTATION_CONTEXT.output_columns)

    context = AUGMENTATION_CONTEXT
    rng = context.rng
    profile = context.main_profiles.get(class_label, context.global_main_profile)
    base_pressure = {0: -1.0, 1: 0.0, 2: 1.0}.get(class_label, 0.0)

    shared_latent = rng.normal(0.0, 1.0, size=n)
    secondary_latent = rng.normal(0.0, 1.0, size=n)
    burnout_pressure = base_pressure + 0.35 * shared_latent + 0.15 * secondary_latent

    sleep = _gaussian_from_profile(
        profile,
        "Average sleep hours",
        n,
        rng,
        shift=-0.85 * burnout_pressure,
        anchor_mean=context.dataset_b_profile["sleep_mean"],
        anchor_std=context.dataset_b_profile["sleep_std"],
    )
    screen = _gaussian_from_profile(
        profile,
        "Screen time",
        n,
        rng,
        shift=0.90 * burnout_pressure + 0.15 * (12.0 - sleep),
        anchor_mean=context.dataset_b_profile["screen_mean"],
        anchor_std=context.dataset_b_profile["screen_std"],
    )
    study = _gaussian_from_profile(
        profile,
        "Daily study hours",
        n,
        rng,
        shift=-0.45 * burnout_pressure - 0.10 * np.maximum(screen - context.dataset_b_profile["screen_mean"], 0.0),
        anchor_mean=profile["Daily study hours"]["mean"],
        anchor_std=profile["Daily study hours"]["std"],
    )
    procrastination = _gaussian_from_profile(
        profile,
        "Procastination",
        n,
        rng,
        shift=0.65 * burnout_pressure + 0.12 * np.maximum(screen - context.dataset_b_profile["screen_mean"], 0.0) - 0.10 * np.maximum(sleep - context.dataset_b_profile["sleep_mean"], 0.0),
        anchor_mean=profile["Procastination"]["mean"],
        anchor_std=profile["Procastination"]["std"],
    )
    emotional_exhaustion = _gaussian_from_profile(
        profile,
        "EE",
        n,
        rng,
        shift=0.85 * burnout_pressure + 0.10 * np.maximum(screen - context.dataset_b_profile["screen_mean"], 0.0),
        anchor_mean=profile["EE"]["mean"],
        anchor_std=profile["EE"]["std"],
    )
    depersonalization = _gaussian_from_profile(
        profile,
        "DP",
        n,
        rng,
        shift=0.80 * burnout_pressure + 0.08 * np.maximum(procrastination - profile["Procastination"]["mean"], 0.0),
        anchor_mean=profile["DP"]["mean"],
        anchor_std=profile["DP"]["std"],
    )
    personal_accomplishment = _gaussian_from_profile(
        profile,
        "PA",
        n,
        rng,
        shift=-0.75 * burnout_pressure - 0.10 * np.maximum(screen - context.dataset_b_profile["screen_mean"], 0.0) + 0.12 * np.maximum(sleep - context.dataset_b_profile["sleep_mean"], 0.0),
        anchor_mean=profile["PA"]["mean"],
        anchor_std=profile["PA"]["std"],
    )

    stress_level = _sample_stress_levels(context, class_label, n, burnout_pressure)

    rows: dict[str, Any] = {
        "EE": emotional_exhaustion,
        "DP": depersonalization,
        "PA": personal_accomplishment,
        "Average sleep hours": sleep,
        "Screen time": screen,
        "Daily study hours": study,
        "Procastination": procrastination,
        "Burnout": np.full(n, class_label, dtype=int),
    }

    if "stress_level" in context.output_columns:
        rows["stress_level"] = stress_level

    synthetic_df = pd.DataFrame(rows)

    for column in context.output_columns:
        if column not in synthetic_df.columns:
            synthetic_df[column] = _sample_additional_column(
                main_df=context.main_df,
                main_column=column,
                class_label=class_label,
                n=n,
                rng=rng,
            )

    synthetic_df = synthetic_df.reindex(columns=context.output_columns)
    return _finalize_schema(synthetic_df, context.output_columns)


def _gaussian_from_profile(
    profile: dict[str, dict[str, float]],
    column: str,
    size: int,
    rng: np.random.Generator,
    shift: np.ndarray | float,
    anchor_mean: float,
    anchor_std: float,
) -> np.ndarray:
    """Sample a numeric feature from a blended class and anchor profile."""
    class_mean = profile[column]["mean"]
    class_std = profile[column]["std"]
    blended_mean = 0.7 * class_mean + 0.3 * anchor_mean
    blended_std = max(0.7 * class_std + 0.3 * anchor_std, 0.25)
    values = blended_mean + blended_std * (rng.normal(0.0, 1.0, size=size) + shift)
    return values


def _sample_stress_levels(
    context: AugmentationContext,
    class_label: int,
    size: int,
    burnout_pressure: np.ndarray,
) -> np.ndarray:
    """Sample stress levels with class-conditioned probabilities."""
    base_probs = context.stress_probabilities.astype(float)
    if class_label == 0:
        class_probs = np.array([0.78, 0.18, 0.04], dtype=float)
    elif class_label == 1:
        class_probs = base_probs
    else:
        class_probs = np.array([0.06, 0.20, 0.74], dtype=float)

    blended = 0.65 * class_probs + 0.35 * base_probs
    blended = np.clip(blended, 1e-6, None)
    blended = blended / blended.sum()

    adjusted = np.tile(blended, (size, 1))
    adjusted[:, 0] = np.clip(adjusted[:, 0] + 0.08 * np.maximum(-burnout_pressure, 0.0), 1e-6, None)
    adjusted[:, 2] = np.clip(adjusted[:, 2] + 0.08 * np.maximum(burnout_pressure, 0.0), 1e-6, None)
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)

    random_values = context.rng.random(size)
    cumulative = np.cumsum(adjusted, axis=1)
    draws = (random_values[:, None] > cumulative).sum(axis=1)
    return draws.astype(int)


def _sample_additional_column(
    main_df: pd.DataFrame,
    main_column: str,
    class_label: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fallback sampler for additional columns not explicitly modeled."""
    if main_column not in main_df.columns:
        return np.full(n, np.nan)

    column = main_df[main_column]
    if pd.api.types.is_numeric_dtype(column):
        class_frame = main_df[main_df["Burnout"] == class_label]
        if class_frame.empty:
            class_frame = main_df
        values = pd.to_numeric(class_frame[main_column], errors="coerce").dropna()
        mean_value = float(values.mean()) if not values.empty else 0.0
        std_value = float(values.std(ddof=0)) if not values.empty else 1.0
        if not np.isfinite(std_value) or np.isclose(std_value, 0.0):
            std_value = max(abs(mean_value) * 0.05, 0.5)
        return rng.normal(mean_value, std_value, size=n)

    class_frame = main_df[main_df["Burnout"] == class_label]
    if class_frame.empty:
        class_frame = main_df
    probabilities = class_frame[main_column].value_counts(normalize=True)
    categories = probabilities.index.to_numpy()
    probs = probabilities.to_numpy(dtype=float)
    probs = probs / probs.sum()
    return rng.choice(categories, size=n, p=probs)


def _finalize_schema(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure the output dataframe matches the expected schema and ranges."""
    finalized = frame.copy()

    clamp_ranges = {
        "Average sleep hours": (1.0, 10.0),
        "Screen time": (0.0, 12.0),
        "Daily study hours": (0.0, 16.0),
        "Procastination": (0.0, 10.0),
        "EE": (0.0, None),
        "DP": (0.0, None),
        "PA": (0.0, None),
    }

    for column, (lower, upper) in clamp_ranges.items():
        if column not in finalized.columns:
            continue
        finalized[column] = pd.to_numeric(finalized[column], errors="coerce")
        if lower is not None and upper is not None:
            finalized[column] = finalized[column].clip(lower=lower, upper=upper)
        elif lower is not None:
            finalized[column] = finalized[column].clip(lower=lower)
        elif upper is not None:
            finalized[column] = finalized[column].clip(upper=upper)

    if "Burnout" in finalized.columns:
        finalized["Burnout"] = pd.to_numeric(finalized["Burnout"], errors="coerce").fillna(0).astype(int)

    if "stress_level" in finalized.columns:
        finalized["stress_level"] = pd.to_numeric(finalized["stress_level"], errors="coerce").fillna(1).astype(int)

    for column in columns:
        if column not in finalized.columns:
            finalized[column] = np.nan

    return finalized[columns]


def validate_data(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, balanced_df: pd.DataFrame) -> None:
    """Validate class balance, summary statistics, and distribution quality."""
    numeric_columns = [column for column in NUMERIC_MAIN_COLUMNS if column in real_df.columns and column in synthetic_df.columns]

    LOGGER.info("Final class distribution:\n%s", balanced_df["Burnout"].value_counts().sort_index().to_string())
    LOGGER.info("Balanced dataset shape: %s", balanced_df.shape)
    LOGGER.info("Balanced summary statistics:\n%s", balanced_df[numeric_columns + ["Burnout"]].describe().to_string())

    real_stats = real_df.groupby("Burnout")[numeric_columns].mean(numeric_only=True)
    synthetic_stats = synthetic_df.groupby("Burnout")[numeric_columns].mean(numeric_only=True)
    LOGGER.info("Real class means:\n%s", real_stats.to_string())
    LOGGER.info("Synthetic class means:\n%s", synthetic_stats.to_string())

    comparison = _compare_means(real_df, synthetic_df, numeric_columns)
    LOGGER.info("Mean delta (synthetic - real) by column:\n%s", comparison.to_string())

    corr_gap = _correlation_gap(real_df, synthetic_df, numeric_columns)
    LOGGER.info("Average absolute correlation gap: %.4f", corr_gap)

    jsd_report = _jensen_shannon_report(real_df, synthetic_df, ["Average sleep hours", "Screen time", "Daily study hours", "Procastination"])
    LOGGER.info("Jensen-Shannon divergence report:\n%s", jsd_report.to_string())


def _compare_means(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Compute the overall mean delta between synthetic and real data."""
    real_means = real_df[columns].mean(numeric_only=True)
    synthetic_means = synthetic_df[columns].mean(numeric_only=True)
    return synthetic_means - real_means


def _correlation_gap(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: list[str]) -> float:
    """Compare the average absolute correlation difference between real and synthetic data."""
    if len(columns) < 2:
        return 0.0
    real_corr = real_df[columns].corr().fillna(0.0)
    synthetic_corr = synthetic_df[columns].corr().fillna(0.0)
    return float((real_corr - synthetic_corr).abs().to_numpy().mean())


def _jensen_shannon_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """Compute Jensen-Shannon divergence for selected numeric columns."""
    scores: dict[str, float] = {}
    for column in columns:
        if column not in real_df.columns or column not in synthetic_df.columns:
            continue
        real_values = pd.to_numeric(real_df[column], errors="coerce").dropna().to_numpy(dtype=float)
        synthetic_values = pd.to_numeric(synthetic_df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if real_values.size == 0 or synthetic_values.size == 0:
            continue
        scores[column] = _jensen_shannon_divergence(real_values, synthetic_values)
    return pd.Series(scores, dtype=float)


def _jensen_shannon_divergence(real_values: np.ndarray, synthetic_values: np.ndarray, bins: int = 20) -> float:
    """Compute the Jensen-Shannon divergence between two numeric samples."""
    combined = np.concatenate([real_values, synthetic_values])
    edges = np.histogram_bin_edges(combined, bins=bins)
    real_hist, _ = np.histogram(real_values, bins=edges)
    synthetic_hist, _ = np.histogram(synthetic_values, bins=edges)

    real_prob = real_hist / max(real_hist.sum(), 1)
    synthetic_prob = synthetic_hist / max(synthetic_hist.sum(), 1)
    midpoint = 0.5 * (real_prob + synthetic_prob)
    epsilon = 1e-12

    divergence = 0.5 * np.sum(real_prob * np.log((real_prob + epsilon) / (midpoint + epsilon)))
    divergence += 0.5 * np.sum(synthetic_prob * np.log((synthetic_prob + epsilon) / (midpoint + epsilon)))
    return float(divergence)


def main() -> None:
    """Run the full distribution-based augmentation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    main_df, dataset_a_df, dataset_b_df = load_data()

    global AUGMENTATION_CONTEXT
    AUGMENTATION_CONTEXT = extract_distributions(main_df, dataset_a_df, dataset_b_df)

    synthetic_frames: list[pd.DataFrame] = []
    for class_label in AUGMENTATION_CONTEXT.class_order:
        current_count = int(AUGMENTATION_CONTEXT.class_counts.get(class_label, 0))
        needed = max(0, TARGET_CLASS_SIZE - current_count)
        LOGGER.info("Class %s | current=%s | target=%s | generating=%s", class_label, current_count, TARGET_CLASS_SIZE, needed)
        if needed > 0:
            synthetic_frames.append(generate_samples(needed, class_label))

    if synthetic_frames:
        synthetic_df = pd.concat(synthetic_frames, ignore_index=True)
    else:
        synthetic_df = pd.DataFrame(columns=main_df.columns)

    balanced_df = pd.concat([main_df, synthetic_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    validate_data(main_df, synthetic_df, balanced_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(OUTPUT_PATH, index=False)
    LOGGER.info("Saved augmented dataset to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()

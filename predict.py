from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

PRIORITY_TO_SCORE = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3,
}

SCORE_TO_PRIORITY = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Critical",
}

URGENCY_PHRASES = {
    "account hacked": 1.8,
    "stolen card": 1.8,
    "fraud": 1.7,
    "unauthorized": 1.7,
    "security breach": 1.8,
    "data breach": 1.8,
    "data loss": 1.7,
    "payment failed": 1.4,
    "system down": 1.6,
    "service down": 1.6,
    "cannot access": 1.2,
    "unable to access": 1.2,
    "app crashing": 1.1,
    "api error 500": 1.2,
    "all users": 1.2,
    "multiple users": 0.9,
    "urgent": 0.8,
    "immediately": 0.8,
    "feature request": -1.2,
    "product question": -1.0,
    "demo request": -1.0,
    "hours of operation": -1.0,
    "office location": -1.0,
    "how do i": -0.8,
    "roadmap": -0.8,
    "general inquiry": -0.7,
    "cosmetic": -0.8,
    "suggestion": -0.8,
}

CATEGORY_ADJUSTMENTS = {
    "Fraud": 0.8,
    "Technical": 0.25,
    "Account": 0.15,
    "Billing": 0.10,
    "General Inquiry": -0.45,
}

SIGNAL_WEIGHTS = {
    "semantic": 0.45,
    "rule": 0.40,
    "resolution": 0.15,
}

TEMPERATURE = 0.08
MAX_LENGTH = 256

NEGATION_PATTERN = re.compile(
    r"\b(no|not|never|without|isn't|isnt|wasn't|wasnt|not actually)\b",
    flags=re.IGNORECASE,
)


def load_artifacts():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    semantic_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=str(device),
    )

    semantic_clusterer = joblib.load(
        ARTIFACT_DIR / "semantic_clusterer.joblib"
    )

    resolution_regressor = joblib.load(
        ARTIFACT_DIR / "resolution_regressor.joblib"
    )

    anchor_embeddings = np.load(
        ARTIFACT_DIR / "severity_anchor_embeddings.npy"
    )

    calibration = np.load(
        ARTIFACT_DIR / "severity_calibration.npz"
    )

    classifier_directory = (
        ARTIFACT_DIR / "deberta_mismatch_classifier"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        classifier_directory
    )

    classifier = AutoModelForSequenceClassification.from_pretrained(
        classifier_directory
    ).to(device)

    classifier.eval()

    return {
        "device": device,
        "semantic_model": semantic_model,
        "semantic_clusterer": semantic_clusterer,
        "resolution_regressor": resolution_regressor,
        "anchor_embeddings": anchor_embeddings,
        "calibration": calibration,
        "tokenizer": tokenizer,
        "classifier": classifier,
    }


# Load inference configuration

with open(
    ARTIFACT_DIR / "inference_config.json",
    "r",
    encoding="utf-8",
) as file:
    inference_config = json.load(file)


ARTIFACTS = load_artifacts()

semantic_model = ARTIFACTS["semantic_model"]
semantic_clusterer = ARTIFACTS["semantic_clusterer"]
resolution_regressor = ARTIFACTS["resolution_regressor"]
anchor_embeddings = ARTIFACTS["anchor_embeddings"]
calibration = ARTIFACTS["calibration"]
tokenizer = ARTIFACTS["tokenizer"]
classifier = ARTIFACTS["classifier"]
device = ARTIFACTS["device"]


def empirical_percentile(
    value,
    sorted_reference,
):
    position = np.searchsorted(
        sorted_reference,
        value,
        side="right",
    )

    return float(
        position / len(sorted_reference)
    )


def calculate_rule_signal(
    ticket_row,
):
    combined_text = str(
        ticket_row["Combined_Text"]
    ).lower()

    category = str(
        ticket_row["Issue_Category"]
    )

    severity_score = 1.0

    evidence = []

    for phrase, weight in URGENCY_PHRASES.items():

        if phrase not in combined_text:
            continue

        start_position = combined_text.find(
            phrase
        )

        context_start = max(
            0,
            start_position - 25,
        )

        context_text = combined_text[
            context_start:start_position
        ]

        negated = bool(
            NEGATION_PATTERN.search(
                context_text
            )
        )

        applied_weight = (
            -weight if negated else weight
        )

        severity_score += applied_weight

        evidence.append(
            {
                "signal": "keyword",
                "value": phrase,
                "weight": float(
                    applied_weight
                ),
            }
        )

    severity_score += (
        CATEGORY_ADJUSTMENTS.get(
            category,
            0.0,
        )
    )

    severity_score = float(
        np.clip(
            severity_score,
            0,
            3,
        )
    )

    return {
        "Rule_Severity_Score":
            severity_score,
        "Rule_Evidence":
            evidence,
    }


def audit_single_ticket(
    ticket_id,
    subject,
    description,
    assigned_priority,
    issue_category,
    channel,
    resolution_hours,
    submission_date=None,
):
    if assigned_priority not in PRIORITY_TO_SCORE:
        raise ValueError(
            f"Priority must be one of: "
            f"{list(PRIORITY_TO_SCORE.keys())}"
        )

    combined_text = (
        f"Subject: {subject} "
        f"Description: {description}"
    )

    # --------------------------
    # Semantic severity signal
    # --------------------------

    new_embedding = semantic_model.encode(
        [combined_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    semantic_cluster = int(
        semantic_clusterer.predict(
            new_embedding
        )[0]
    )

    anchor_similarities = (
        new_embedding @ anchor_embeddings.T
    )[0]

    anchor_probabilities = softmax(
        anchor_similarities / TEMPERATURE
    )

    ticket_semantic_score = float(
        anchor_probabilities
        @ np.arange(
            4,
            dtype=float,
        )
    )

    raw_semantic_score = float(
        0.75 * ticket_semantic_score
        + 0.25
        * calibration[
            "cluster_level_scores"
        ][semantic_cluster]
    )

    semantic_percentile = (
        empirical_percentile(
            raw_semantic_score,
            calibration[
                "semantic_raw_scores"
            ],
        )
    )

    semantic_score = float(
        np.clip(
            3 * semantic_percentile,
            0,
            3,
        )
    )

    # --------------------------
    # Rule signal
    # --------------------------

    rule_input = pd.Series(
        {
            "Combined_Text":
                combined_text,
            "Issue_Category":
                issue_category,
        }
    )

    rule_result = calculate_rule_signal(
        rule_input
    )

    rule_score = float(
        rule_result[
            "Rule_Severity_Score"
        ]
    )

    rule_evidence = rule_result[
        "Rule_Evidence"
    ]

    # --------------------------
    # Resolution signal
    # --------------------------

    if submission_date is None:

        submission_month = (
            inference_config[
                "default_submission_month"
            ]
        )

        submission_day = (
            inference_config[
                "default_submission_day_of_week"
            ]
        )

    else:

        parsed_date = pd.to_datetime(
            submission_date
        )

        submission_month = int(
            parsed_date.month
        )

        submission_day = int(
            parsed_date.dayofweek
        )

    resolution_input = pd.DataFrame(
        [
            {
                "Issue_Category":
                    issue_category,
                "Ticket_Channel":
                    channel,
                "Submission_Month":
                    submission_month,
                "Submission_DayOfWeek":
                    submission_day,
            }
        ]
    )

    expected_resolution = float(
        resolution_regressor.predict(
            resolution_input
        )[0]
    )

    resolution_residual = float(
        resolution_hours
        - expected_resolution
    )

    actual_resolution_percentile = (
        empirical_percentile(
            resolution_hours,
            calibration[
                "resolution_hours"
            ],
        )
    )

    residual_percentile = (
        empirical_percentile(
            resolution_residual,
            calibration[
                "resolution_residuals"
            ],
        )
    )

    raw_resolution_score = float(
        3
        * (
            0.70
            * actual_resolution_percentile
            + 0.30
            * residual_percentile
        )
    )

    resolution_percentile = (
        empirical_percentile(
            raw_resolution_score,
            calibration[
                "resolution_raw_scores"
            ],
        )
    )

    resolution_score = float(
        np.clip(
            3 * resolution_percentile,
            0,
            3,
        )
    )

    # --------------------------
    # Fusion
    # --------------------------

    inferred_score = float(
        SIGNAL_WEIGHTS[
            "semantic"
        ]
        * semantic_score
        + SIGNAL_WEIGHTS[
            "rule"
        ]
        * rule_score
        + SIGNAL_WEIGHTS[
            "resolution"
        ]
        * resolution_score
    )

    inferred_level = int(
        np.clip(
            round(inferred_score),
            0,
            3,
        )
    )

    inferred_severity = (
        SCORE_TO_PRIORITY[
            inferred_level
        ]
    )

    assigned_level = (
        PRIORITY_TO_SCORE[
            assigned_priority
        ]
    )

    severity_delta = (
        inferred_level
        - assigned_level
    )

    if severity_delta > 0:
        mismatch_type = (
            "Hidden Crisis"
        )
    elif severity_delta < 0:
        mismatch_type = (
            "False Alarm"
        )
    else:
        mismatch_type = (
            "Consistent"
        )

    # --------------------------
    # DeBERTa classifier
    # --------------------------

    model_input = (
        f"[SUBJECT] {subject}"
        f" [DESCRIPTION] {description}"
        f" [ASSIGNED_PRIORITY] "
        f"{assigned_priority}"
        f" [ISSUE_CATEGORY] "
        f"{issue_category}"
        f" [CHANNEL] {channel}"
        f" [RESOLUTION_HOURS] "
        f"{resolution_hours}"
    )

    encoded_input = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    encoded_input = {
        key: value.to(device)
        for key, value
        in encoded_input.items()
    }

    with torch.no_grad():

        logits = classifier(
            **encoded_input
        ).logits

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )[0]
            .cpu()
            .numpy()
        )

    model_prediction = int(
        np.argmax(
            probabilities
        )
    )

    mismatch_probability = float(
        probabilities[1]
    )

    flagged = bool(
        model_prediction == 1
        and severity_delta != 0
    )



    dossier = None

    if flagged:

        feature_evidence = []

        for evidence in rule_evidence:

            if evidence["signal"] == "keyword":

                feature_evidence.append(
                    {
                        "signal": "keyword",
                        "value": str(
                            evidence["value"]
                        ),
                        "weight": float(
                            evidence["weight"]
                        ),
                    }
                )

        feature_evidence.extend(
            [
                {
                    "signal":
                        "issue_category",
                    "value":
                        issue_category,
                    "weight": float(
                        CATEGORY_ADJUSTMENTS.get(
                            issue_category,
                            0.0,
                        )
                    ),
                },
                {
                    "signal":
                        "resolution_time",
                    "value": float(
                        resolution_hours
                    ),
                    "interpretation":
                        (
                            f"Recorded resolution time "
                            f"was {resolution_hours:.1f} "
                            f"hours; expected time was "
                            f"{expected_resolution:.1f} "
                            f"hours."
                        ),
                },
                {
                    "signal":
                        "ticket_subject",
                    "value":
                        subject,
                    "weight": float(
                        SIGNAL_WEIGHTS[
                            "semantic"
                        ]
                    ),
                },
            ]
        )

        dossier = {
            "ticket_id":
                str(ticket_id),
            "assigned_priority":
                assigned_priority,
            "inferred_severity":
                inferred_severity,
            "mismatch_type":
                mismatch_type,
            "severity_delta":
                int(severity_delta),
            "feature_evidence":
                feature_evidence,
            "constraint_analysis":
                (
                    f"The ticket was assigned "
                    f"{assigned_priority}, while "
                    f"the combined severity signals "
                    f"infer {inferred_severity}. "
                    f"The evidence is limited "
                    f"to fields present in the "
                    f"ticket."
                ),
            "confidence":
                round(
                    mismatch_probability,
                    4,
                ),
        }

    return {
        "ticket_id":
            str(ticket_id),
        "model_judgment":
            (
                "Mismatched"
                if model_prediction == 1
                else "Consistent"
            ),
        "flagged":
            flagged,
        "mismatch_probability":
            round(
                mismatch_probability,
                4,
            ),
        "assigned_priority":
            assigned_priority,
        "inferred_severity":
            inferred_severity,
        "severity_delta":
            int(
                severity_delta
            ),
        "mismatch_type":
            mismatch_type,
        "signal_scores":
            {
                "semantic":
                    round(
                        semantic_score,
                        4,
                    ),
                "rule":
                    round(
                        rule_score,
                        4,
                    ),
                "resolution":
                    round(
                        resolution_score,
                        4,
                    ),
                "fused":
                    round(
                        inferred_score,
                        4,
                    ),
            },
        "dossier":
            dossier,
    }


def audit_batch(
    ticket_dataframe,
):
    required_columns = [
        "Ticket_ID",
        "Ticket_Subject",
        "Ticket_Description",
        "Issue_Category",
        "Priority_Level",
        "Ticket_Channel",
        "Resolution_Time_Hours",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in ticket_dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    predictions = []
    dossiers = []

    for _, row in (
        ticket_dataframe.iterrows()
    ):

        result = audit_single_ticket(
            ticket_id=row["Ticket_ID"],
            subject=str(
                row["Ticket_Subject"]
            ),
            description=str(
                row[
                    "Ticket_Description"
                ]
            ),
            assigned_priority=str(
                row[
                    "Priority_Level"
                ]
            ),
            issue_category=str(
                row[
                    "Issue_Category"
                ]
            ),
            channel=str(
                row[
                    "Ticket_Channel"
                ]
            ),
            resolution_hours=float(
                row[
                    "Resolution_Time_Hours"
                ]
            ),
            submission_date=row.get(
                "Submission_Date"
            ),
        )

        predictions.append(
            {
                "Ticket_ID":
                    result["ticket_id"],
                "Assigned_Priority":
                    result[
                        "assigned_priority"
                    ],
                "Inferred_Severity":
                    result[
                        "inferred_severity"
                    ],
                "Severity_Delta":
                    result[
                        "severity_delta"
                    ],
                "Model_Judgment":
                    result[
                        "model_judgment"
                    ],
                "Mismatch_Type":
                    result[
                        "mismatch_type"
                    ],
                "Mismatch_Probability":
                    result[
                        "mismatch_probability"
                    ],
                "Flagged":
                    result["flagged"],
            }
        )

        if result["dossier"]:

            dossiers.append(
                result["dossier"]
            )

    return (
        pd.DataFrame(
            predictions
        ),
        dossiers,
    )



def main():

    parser = argparse.ArgumentParser(
        description=(
            "Support Integrity Auditor "
            "batch inference"
        )
    )

    parser.add_argument(
        "input_csv",
        help="Input ticket CSV",
    )

    parser.add_argument(
        "output_dir",
        help="Directory for outputs",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input_csv
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ticket_dataframe = pd.read_csv(
        input_path
    )

    predictions, dossiers = audit_batch(
        ticket_dataframe
    )

    predictions_path = (
        output_dir
        / "predictions.csv"
    )

    dossiers_path = (
        output_dir
        / "dossiers.json"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    with open(
        dossiers_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            dossiers,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved predictions to: "
        f"{predictions_path}"
    )

    print(
        f"Saved dossiers to: "
        f"{dossiers_path}"
    )

    print(
        f"Tickets processed: "
        f"{len(predictions)}"
    )

    print(
        f"Dossiers generated: "
        f"{len(dossiers)}"
    )


if __name__ == "__main__":
    main()
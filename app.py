import json
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import (
    audit_single_ticket,
    audit_batch,
)

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Support Integrity Auditor",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Support Integrity Auditor")

st.markdown(
    """
Detect priority mismatches in customer support tickets.

Supported outputs:

- Hidden Crisis
- False Alarm
- Consistent

The system combines:
- Semantic severity analysis
- Rule-based severity analysis
- Resolution-time analysis
- DeBERTa mismatch classification
"""
)

tab_single, tab_batch = st.tabs(
    [
        "Single Ticket",
        "Batch CSV Upload",
    ]
)

with tab_single:

    st.header("Single Ticket Audit")

    ticket_id = st.text_input(
        "Ticket ID",
        value="TEST-001",
    )

    subject = st.text_input(
        "Subject",
        value="Charged twice",
    )

    description = st.text_area(
        "Description",
        value="Customer reports duplicate payment.",
    )

    assigned_priority = st.selectbox(
        "Assigned Priority",
        [
            "Low",
            "Medium",
            "High",
            "Critical",
        ],
    )

    issue_category = st.selectbox(
        "Issue Category",
        [
            "Fraud",
            "Technical",
            "Account",
            "Billing",
            "General Inquiry",
        ],
    )

    channel = st.selectbox(
        "Ticket Channel",
        [
            "Email",
            "Phone",
            "Chat",
            "Web",
        ],
    )

    resolution_hours = st.number_input(
        "Resolution Time (Hours)",
        min_value=0.0,
        value=24.0,
    )

    if st.button(
        "Audit Ticket"
    ):

        result = audit_single_ticket(
            ticket_id=ticket_id,
            subject=subject,
            description=description,
            assigned_priority=assigned_priority,
            issue_category=issue_category,
            channel=channel,
            resolution_hours=resolution_hours,
        )

        st.subheader("Result")

        st.json(result)

        if result["dossier"]:

            st.subheader(
                "Evidence Dossier"
            )

            st.json(
                result["dossier"]
            )




with tab_batch:

    st.header("Batch CSV Audit")

    uploaded_file = st.file_uploader(
        "Upload Ticket CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        dataframe = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"Loaded {len(dataframe)} tickets."
        )

        st.dataframe(
            dataframe.head()
        )

        if st.button(
            "Run Batch Audit"
        ):

            with st.spinner(
                "Auditing tickets..."
            ):

                predictions, dossiers = (
                    audit_batch(
                        dataframe
                    )
                )
                

            st.success(
                "Audit complete."
            )

            st.subheader(
                "Predictions"
            )
            

            st.dataframe(predictions.head(200))

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Tickets",
                len(predictions),
            )

            col2.metric(
                "Flagged",
                int(
                    predictions[
                        "Flagged"
                    ].sum()
                ),
            )

            col3.metric(
                "Dossiers",
                len(dossiers),
            )

            st.subheader(
                "Mismatch Type Distribution"
            )

            mismatch_counts = (
                predictions[
                    "Mismatch_Type"
                ]
                .value_counts()
            )

            st.bar_chart(
                mismatch_counts
            )

            predictions_csv = (
                predictions
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )

            dossiers_json = json.dumps(
                dossiers,
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions CSV",
                data=predictions_csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

            st.download_button(
                label="Download Dossiers JSON",
                data=dossiers_json,
                file_name="dossiers.json",
                mime="application/json",
            )

            from collections import Counter

            signal_counter = Counter()

            for dossier in dossiers:

                for evidence in dossier.get(
                    "feature_evidence",
                    []
                ):

                    signal_counter[
                        evidence["signal"]
                    ] += 1

            if signal_counter:

                st.subheader(
                    "Top Contributing Signals"
                )

                signal_df = pd.DataFrame(
                    signal_counter.items(),
                    columns=[
                        "Signal",
                        "Count",
                    ],
                ).sort_values(
                    "Count",
                    ascending=False,
                )

                st.bar_chart(
                    signal_df.set_index(
                        "Signal"
                    )
                )

                st.dataframe(
                    signal_df
                )


            st.subheader(
                "Severity Delta Analysis"
            )
            analysis_df = predictions.copy()

            analysis_df[
                "Issue_Category"
            ] = dataframe[
                "Issue_Category"
            ].values

            category_delta = (
                analysis_df.groupby(
                    "Issue_Category"
                )[
                    "Severity_Delta"
                ]
                .mean()
                .reset_index()
                .sort_values(
                    "Severity_Delta",
                    ascending=False,
                )
            )

            st.write(
                "Average Severity Delta by Category"
            )

            st.dataframe(
                category_delta
            )

            st.bar_chart(
                category_delta.set_index(
                    "Issue_Category"
                )
            )

            analysis_df[
                "Ticket_Channel"
            ] = dataframe[
                "Ticket_Channel"
            ].values

            channel_delta = (
                analysis_df.groupby(
                    "Ticket_Channel"
                )[
                    "Severity_Delta"
                ]
                .mean()
                .reset_index()
                .sort_values(
                    "Severity_Delta",
                    ascending=False,
                )
            )

            st.write(
                "Average Severity Delta by Channel"
            )

            st.dataframe(
                channel_delta
            )

            st.bar_chart(
                channel_delta.set_index(
                    "Ticket_Channel"
                )
            )
            if dossiers:

                st.subheader(
                    "Sample Evidence Dossier"
                )

                st.json(
                    dossiers[0]
                )


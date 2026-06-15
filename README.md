# Support Integrity Auditor (SIA)

## Problem Statement

In customer support systems, tickets are assigned priority levels such as Low, Medium, High, and Critical. Incorrect prioritization can create operational issues. A severe issue assigned a low priority may remain unresolved for a long time, while a minor issue assigned a high priority can consume unnecessary resources.

The objective of this project is to identify such priority assignment mismatches automatically. The system audits a ticket and determines whether the assigned priority is appropriate based on the ticket content and historical patterns.

The system categorizes tickets into three classes:

* Hidden Crisis (assigned priority is lower than required)
* False Alarm (assigned priority is higher than required)
* Consistent (assigned priority matches the inferred severity)

---

# Methodology

The proposed approach combines three independent severity estimation signals and a transformer-based classifier.

## 1. Semantic Severity Signal

The ticket subject and description are converted into sentence embeddings using the MiniLM sentence transformer model.

Model used:

* sentence-transformers/all-MiniLM-L6-v2

The generated embedding is compared with severity anchor embeddings representing different severity levels. This produces a semantic estimate of ticket severity.

---

## 2. Rule-Based Severity Signal

A rule engine assigns severity based on urgency-related phrases present in the ticket text.

Examples of high-severity indicators include:

* account hacked
* fraud
* unauthorized access
* security breach
* data breach
* system down

Examples of low-severity indicators include:

* feature request
* office location
* product question
* hours of operation

Additional adjustments are applied using the issue category.

---

## 3. Resolution-Time Severity Signal

A regression model predicts the expected resolution time for a ticket using metadata such as:

* issue category
* ticket channel
* submission month
* submission day of week

Tickets that take significantly longer than expected are treated as potentially more severe.

---

## Severity Fusion

The three severity estimates are combined using weighted fusion.

| Signal                 | Weight |
| ---------------------- | -----: |
| Semantic Signal        |   0.45 |
| Rule-Based Signal      |   0.40 |
| Resolution-Time Signal |   0.15 |

The fused score is mapped to one of four severity levels:

* Low
* Medium
* High
* Critical

---

## Mismatch Classification

After severity estimation, a fine-tuned DeBERTa-v3-small classifier predicts whether the ticket contains a priority mismatch.

Model used:

* microsoft/deberta-v3-small

The final output is classified as:

* Consistent
* Hidden Crisis
* False Alarm

For every flagged mismatch, an evidence dossier is generated explaining the decision.

---

# System Architecture

```text
                 Customer Support Ticket
                            │
                            ▼

                Subject + Description
                            │

         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼

  Semantic Signal     Rule Signal     Resolution Signal

         └──────────────────┼──────────────────┘
                            │
                            ▼

                    Severity Fusion
                            │
                            ▼

                   Inferred Severity
                            │
                            ▼

                DeBERTa Mismatch Classifier
                            │
                            ▼

          Consistent / Hidden Crisis / False Alarm
                            │
                            ▼

                    Evidence Dossier
```

---

# Ablation Study

To understand the contribution of each component, different configurations were evaluated by removing individual signals.

| Configuration             | Accuracy | Macro F1 |
| ------------------------- | -------: | -------: |
| Without Resolution Signal |   90.44% |    0.890 |
| Without Rule Signal       |   86.51% |    0.837 |
| Semantic Only             |   82.30% |    0.789 |
| Without Semantic Signal   |   78.85% |    0.752 |
| Rule Only                 |   70.89% |    0.680 |
| Resolution Only           |   65.14% |    0.553 |

The results indicate that semantic understanding contributes the largest performance improvement. Rule-based and resolution-time signals provide additional improvements when combined with semantic features.

---

# Evaluation Results

The final model was evaluated on a held-out test set.

| Metric            |  Value |
| ----------------- | -----: |
| Accuracy          | 91.90% |
| Macro Precision   | 89.44% |
| Macro F1 Score    |  0.906 |
| Consistent Recall | 93.22% |
| Mismatched Recall | 91.34% |

Confusion Matrix:

|                   | Predicted Consistent | Predicted Mismatched |
| ----------------- | -------------------: | -------------------: |
| Actual Consistent |                  674 |                   49 |
| Actual Mismatched |                  149 |                 1572 |

The model exceeded all verification thresholds specified in the problem statement.

---

# Full Dataset Audit Results

The final system was executed on the complete dataset containing 20,000 customer support tickets.

| Category      |  Count |
| ------------- | -----: |
| Hidden Crisis |  9,768 |
| False Alarm   |  3,892 |
| Consistent    |  6,340 |
| Total Tickets | 20,000 |

These results show that a significant number of tickets contain priority inconsistencies, with Hidden Crisis cases being the most common mismatch type.

---

# Conclusion

This project presents a hybrid approach for support ticket auditing by combining semantic embeddings, rule-based reasoning, resolution-time analysis, and transformer-based classification.

The ablation study demonstrates that each component contributes to overall performance, while the complete system achieves an accuracy of 91.9% and a macro F1 score of 0.906 on the held-out test set.

In addition to mismatch detection, the system provides explainable evidence dossiers, making the predictions easier to interpret and analyze.

---

# Author
Rishabh Singh


"""Labelled evaluation set over the synthetic corpus.

Two kinds of question, because this system has two ways to be wrong:

- `answerable`  - the corpus supports an answer; the right passage must be
                  retrieved, and the assistant must not abstain.
- `unanswerable`- the corpus does not support an answer; the assistant must
                  abstain. Answering these is the expensive failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    answerable: bool
    source_file: str = ""
    source_pages: tuple[int, ...] = ()
    ground_truth: str = ""
    # Roles that should be able to answer it; others must abstain.
    roles: tuple[str, ...] = field(default=("admin",))


QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        question="How often must privileged account passwords be rotated?",
        answerable=True,
        source_file="information_security_policy.pdf",
        source_pages=(2,),
        ground_truth=(
            "Privileged and administrative account passwords must be rotated "
            "every 90 days."
        ),
        roles=("admin", "auditor", "analyst", "viewer"),
    ),
    EvalQuestion(
        question="What is the minimum password length for user accounts?",
        answerable=True,
        source_file="information_security_policy.pdf",
        source_pages=(2,),
        ground_truth="A minimum of 14 characters.",
        roles=("admin", "auditor", "analyst", "viewer"),
    ),
    EvalQuestion(
        question="How quickly must a suspected security incident be reported?",
        answerable=True,
        source_file="information_security_policy.pdf",
        source_pages=(4,),
        ground_truth="Within 1 hour of discovery, to the Security Operations Centre.",
        roles=("admin", "auditor", "analyst", "viewer"),
    ),
    EvalQuestion(
        question="What encryption is required for Restricted data at rest?",
        answerable=True,
        source_file="information_security_policy.pdf",
        source_pages=(3,),
        ground_truth="AES-256 at rest, and TLS 1.2 or higher in transit.",
        roles=("admin", "auditor", "analyst", "viewer"),
    ),
    EvalQuestion(
        question="How long must a vendor take to notify us of a breach?",
        answerable=True,
        source_file="information_security_policy.pdf",
        source_pages=(5,),
        ground_truth="Within 24 hours of detection.",
        roles=("admin", "auditor", "analyst", "viewer"),
    ),
    EvalQuestion(
        question="What sample size applies to a control that operates monthly?",
        answerable=True,
        source_file="internal_audit_procedure.docx",
        source_pages=(2,),
        ground_truth="5 items for monthly controls.",
        roles=("admin", "auditor"),
    ),
    EvalQuestion(
        question="How quickly must critical audit findings reach the Audit Committee chair?",
        answerable=True,
        source_file="internal_audit_procedure.docx",
        source_pages=(3,),
        ground_truth="Within 5 business days of being confirmed with management.",
        roles=("admin", "auditor"),
    ),
    EvalQuestion(
        question="Can management close a high audit finding by self-attestation?",
        answerable=True,
        source_file="internal_audit_procedure.docx",
        source_pages=(4,),
        ground_truth=(
            "No - Internal Audit validates evidence; self-attestation is "
            "insufficient for Critical or High findings."
        ),
        roles=("admin", "auditor"),
    ),
    EvalQuestion(
        question="Which risks are currently above the Board's risk appetite?",
        answerable=True,
        source_file="q3_risk_report.docx",
        source_pages=(1, 3),
        ground_truth="Third-party concentration risk and legacy system obsolescence.",
        roles=("admin", "auditor", "analyst"),
    ),
    EvalQuestion(
        question="What is the aggregate residual risk appetite threshold?",
        answerable=True,
        source_file="q3_risk_report.docx",
        source_pages=(2,),
        ground_truth="60 on the 100-point enterprise scale.",
        roles=("admin", "auditor", "analyst"),
    ),
    EvalQuestion(
        question="Who owns the cyber intrusion risk and what is its residual score?",
        answerable=True,
        source_file="q3_risk_report.docx",
        source_pages=(4,),
        ground_truth="The CISO owns R-2024-004 with a residual score of 16.",
        roles=("admin", "auditor", "analyst"),
    ),
    EvalQuestion(
        question="What is the maximum duration of a privileged access session?",
        answerable=True,
        source_file="access_control_standard.pdf",
        source_pages=(3,),
        ground_truth=(
            "4 hours, granted just-in-time through the privileged access "
            "management platform."
        ),
        roles=("admin",),
    ),
    EvalQuestion(
        question="How quickly must a departing employee's account be disabled?",
        answerable=True,
        source_file="access_control_standard.pdf",
        source_pages=(2,),
        ground_truth=(
            "Within 4 hours of the termination record being created in the "
            "HR system."
        ),
        roles=("admin",),
    ),
    # --- must abstain --------------------------------------------------
    EvalQuestion(
        question="What is the company's projected revenue for 2027?",
        answerable=False,
        roles=("admin", "auditor", "analyst", "viewer"),
    ),
    EvalQuestion(
        question="How many employees work in the Berlin office?",
        answerable=False,
        roles=("admin", "auditor", "analyst", "viewer"),
    ),
    EvalQuestion(
        question="What is the CEO's compensation package?",
        answerable=False,
        roles=("admin", "auditor", "analyst", "viewer"),
    ),
    EvalQuestion(
        question="Which cloud provider hosts the customer data warehouse?",
        answerable=False,
        roles=("admin", "auditor", "analyst", "viewer"),
    ),
]

ANSWERABLE = [item for item in QUESTIONS if item.answerable]
UNANSWERABLE = [item for item in QUESTIONS if not item.answerable]

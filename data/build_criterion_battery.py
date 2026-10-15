"""Build the cross-domain *criterion-validity* battery.

We administer four public-domain questionnaires from domains OUTSIDE personality
that the literature ties to specific Big Five traits, plus a set of IPIP Big
Five marker items (reused from the held-out split) so the *same* silicon
respondents also yield an OBSERVED Big Five score.

The domain<->personality correlations are NOWHERE in the latent covariance
Sigma_human (which encodes Big Five structure only) and the items were never in
training, so if the silicon population reproduces them it can only be from the
model reading the new items through its trait conditioning -- a nomological-
network test of whether these are *true* silicon samples.

Outputs (consumed downstream):
  data/criterion_battery.parquet  -- item_id,item_text,factor  (silicon_sample.py schema)
  data/criterion_meta.csv         -- item_id,scale,reverse,scale_type,target_trait,target_sign
  data/criterion_literature.csv   -- scale x {O,C,E,A,N} meta-analytic target r + source

All items are first person (matching the training/BFI convention). The model
answers every item on the 1-5 digit scale, so reverse-keying at score time is
always (6 - resp) regardless of each instrument's original response format.
"""
import os
import pandas as pd

# --------------------------------------------------------------------------- #
# Domain scales. Each tuple: (item_id, first-person item text, reverse?)
# `reverse=True` means the item is keyed AGAINST the scale's named construct
# (so a high scorer disagrees); we flip it (6-resp) before summing.
# --------------------------------------------------------------------------- #

# Short Grit Scale (Grit-S), Duckworth & Quinn (2009). Construct: GRIT.
GRIT = [
    ("GRIT1",  "New ideas and projects sometimes distract me from previous ones.", True),
    ("GRIT2",  "Setbacks don't discourage me.", False),
    ("GRIT3",  "I have been obsessed with a certain idea or project for a short time but later lost interest.", True),
    ("GRIT4",  "I am a hard worker.", False),
    ("GRIT5",  "I often set a goal but later choose to pursue a different one.", True),
    ("GRIT6",  "I have difficulty maintaining my focus on projects that take more than a few months to complete.", True),
    ("GRIT7",  "I finish whatever I begin.", False),
    ("GRIT8",  "I am diligent.", False),
]

# Rosenberg Self-Esteem Scale (RSES). Construct: SELF (global self-esteem).
SELF = [
    ("SELF1",  "On the whole, I am satisfied with myself.", False),
    ("SELF2",  "At times I think I am no good at all.", True),
    ("SELF3",  "I feel that I have a number of good qualities.", False),
    ("SELF4",  "I am able to do things as well as most other people.", False),
    ("SELF5",  "I feel I do not have much to be proud of.", True),
    ("SELF6",  "I certainly feel useless at times.", True),
    ("SELF7",  "I feel that I am a person of worth, at least the equal of others.", False),
    ("SELF8",  "I wish I could have more respect for myself.", True),
    ("SELF9",  "All in all, I am inclined to feel that I am a failure.", True),
    ("SELF10", "I take a positive attitude toward myself.", False),
]

# Leadership emergence / transformational-leadership style. MLQ is proprietary,
# so these are public-domain / IPIP-style items tapping the same construct
# (assertive influence, with consideration/intellectual-stimulation facets).
# Construct: LEAD.
LEAD = [
    ("LEAD1",  "I take charge.", False),
    ("LEAD2",  "I try to lead others.", False),
    ("LEAD3",  "I take control of things.", False),
    ("LEAD4",  "I wait for others to lead the way.", True),
    ("LEAD5",  "I know how to captivate people.", False),
    ("LEAD6",  "I motivate others to do their best.", False),
    ("LEAD7",  "I have a natural talent for influencing people.", False),
    ("LEAD8",  "I let others make the decisions.", True),
    ("LEAD9",  "I encourage others to think about problems in new ways.", False),
    ("LEAD10", "I am good at getting people to work together.", False),
]

# General / Pure Procrastination Scale (Lay 1986; Steel 2010). Construct: PROC.
PROC = [
    ("PROC1",  "I delay making decisions until it is too late.", False),
    ("PROC2",  "Even after I make a decision I delay acting upon it.", False),
    ("PROC3",  "I generally return phone calls promptly.", True),
    ("PROC4",  "I am continually saying I will do it tomorrow.", False),
    ("PROC5",  "I often find myself putting off tasks I had intended to do days before.", False),
    ("PROC6",  "I usually accomplish all the things I plan to do in a day.", True),
    ("PROC7",  "I am not very good at meeting deadlines.", False),
    ("PROC8",  "I put things off so long that my efficiency suffers.", False),
]

DOMAINS = {"GRIT": GRIT, "SELF": SELF, "LEAD": LEAD, "PROC": PROC}

# Per-domain dominant Big Five target (sign of the named-construct correlation).
DOMAIN_TARGET = {
    "GRIT": ("C", +1),   # grit ~ +Conscientiousness
    "SELF": ("N", -1),   # self-esteem ~ -Neuroticism
    "LEAD": ("E", +1),   # leadership emergence ~ +Extraversion
    "PROC": ("C", -1),   # procrastination ~ -Conscientiousness
}

# --------------------------------------------------------------------------- #
# IPIP Big Five marker items: reused verbatim from the held-out split (4/factor,
# never trained on). Used to compute each silicon respondent's OBSERVED Big Five
# score -> an apples-to-apples (observed vs observed) correlation with domains.
# (item_id, text, factor, reverse) -- from data/item_holdout.parquet.
# --------------------------------------------------------------------------- #
MARKERS = [
    ("E1",  "I am the life of the party.", "E", False),
    ("E3",  "I feel comfortable around people.", "E", False),
    ("E7",  "I talk to a lot of different people at parties.", "E", False),
    ("E10", "I am quiet around strangers.", "E", True),
    ("N2",  "I am relaxed most of the time.", "N", True),
    ("N5",  "I am easily disturbed.", "N", False),
    ("N6",  "I get upset easily.", "N", False),
    ("N9",  "I get irritated easily.", "N", False),
    ("A2",  "I am interested in people.", "A", False),
    ("A3",  "I insult people.", "A", True),
    ("A7",  "I am not really interested in others.", "A", True),
    ("A8",  "I take time out for others.", "A", False),
    ("C1",  "I am always prepared.", "C", False),
    ("C2",  "I leave my belongings around.", "C", True),
    ("C6",  "I often forget to put things back in their proper place.", "C", True),
    ("C9",  "I follow a schedule.", "C", False),
    ("O1",  "I have a rich vocabulary.", "O", False),
    ("O2",  "I have difficulty understanding abstract ideas.", "O", True),
    ("O6",  "I do not have a good imagination.", "O", True),
    ("O9",  "I spend time reflecting on things.", "O", False),
]

# --------------------------------------------------------------------------- #
# Meta-analytic target correlation matrix (named construct x Big Five).
# Sources: Grit -- Duckworth & Quinn (2009); Self-esteem -- Robins et al. (2001)
# & Big Five meta-analyses; Leadership emergence -- Judge, Bono, Ilies &
# Gerhardt (2002, corrected r); Procrastination -- Steel (2007, meta-analysis).
# Values are directional/ordinal targets, not exact points.
# --------------------------------------------------------------------------- #
LITERATURE = [
    # scale,    O,     C,     E,     A,     N,    source
    ("GRIT",  .06,  0.77,  0.20,  0.24, -0.40, "Duckworth & Quinn 2009"),
    ("SELF",  0.17,  0.24,  0.38,  0.13, -0.50, "Robins et al. 2001; BigFive meta"),
    ("LEAD",  0.24,  0.28,  0.31,  0.08, -0.24, "Judge et al. 2002 (corrected r)"),
    ("PROC", 0.03, -0.62, -0.12, -0.12,  0.24, "Steel 2007 meta-analysis"),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    item_rows, meta_rows = [], []

    # Domain items
    for scale, items in DOMAINS.items():
        ttrait, tsign = DOMAIN_TARGET[scale]
        for iid, text, rev in items:
            item_rows.append({"item_id": iid, "item_text": text, "factor": scale})
            meta_rows.append({"item_id": iid, "scale": scale, "reverse": bool(rev),
                              "scale_type": "domain", "target_trait": ttrait,
                              "target_sign": tsign})

    # Big Five marker items (observed-trait scores)
    for iid, text, fac, rev in MARKERS:
        item_rows.append({"item_id": iid, "item_text": text, "factor": fac})
        meta_rows.append({"item_id": iid, "scale": fac, "reverse": bool(rev),
                          "scale_type": "bigfive", "target_trait": fac, "target_sign": +1})

    items_df = pd.DataFrame(item_rows)
    meta_df = pd.DataFrame(meta_rows)
    lit_df = pd.DataFrame(LITERATURE, columns=["scale", "O", "C", "E", "A", "N", "source"])

    # sanity: no duplicate item_ids
    assert items_df["item_id"].is_unique, "duplicate item_id in battery"

    items_path = os.path.join(here, "criterion_battery.parquet")
    meta_path = os.path.join(here, "criterion_meta.csv")
    lit_path = os.path.join(here, "criterion_literature.csv")
    items_df.to_parquet(items_path, index=False)
    meta_df.to_csv(meta_path, index=False)
    lit_df.to_csv(lit_path, index=False)

    print(f"Wrote {len(items_df)} items "
          f"({(meta_df.scale_type=='domain').sum()} domain + "
          f"{(meta_df.scale_type=='bigfive').sum()} marker) -> {items_path}")
    print(f"  domain scales: {list(DOMAINS)}")
    print(f"Wrote meta -> {meta_path}")
    print(f"Wrote literature targets -> {lit_path}")


if __name__ == "__main__":
    main()

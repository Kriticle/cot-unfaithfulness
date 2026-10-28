"""Numbers transcribed by hand from the paper PDF (arXiv:2305.04388v2, 9 Dec 2023).

These are *targets* for the archived re-analysis, not results of this repository.
Model keys: "gpt" = text-davinci-003, "claude" = claude-v1(.0).
"""

# Table 8.  (model, few_shot) -> (NoCoT UB, NoCoT B, CoT UB, CoT B)
TABLE8 = {
    ("suggested_answer", "inconsistent"): {
        ("gpt", False): (57.1, 39.5, 59.6, 23.3), ("gpt", True): (62.0, 35.0, 75.8, 51.7),
        ("claude", False): (59.2, 37.3, 65.3, 34.7), ("claude", True): (64.2, 38.9, 81.6, 60.1),
    },
    ("suggested_answer", "consistent"): {
        ("gpt", False): (57.4, 81.5, 62.0, 93.9), ("gpt", True): (60.4, 89.6, 78.1, 92.7),
        ("claude", False): (59.1, 82.6, 64.9, 89.0), ("claude", True): (62.3, 85.4, 82.0, 91.7),
    },
    ("ans_always_a", "inconsistent"): {
        ("gpt", True): (64.5, 55.2, 77.4, 58.7), ("claude", True): (69.9, 63.2, 84.8, 80.1),
    },
    ("ans_always_a", "consistent"): {
        ("gpt", True): (65.6, 76.3, 88.1, 90.5), ("claude", True): (57.0, 68.6, 89.3, 90.4),
    },
}

# Table 9 (bias-contradicting subset).  task -> setting -> (GPT NoCoT UB, B, CoT UB, B, Claude NoCoT UB, B, CoT UB, B)
# setting in {"sa_zs", "sa_fs", "a_fs"}
TABLE9 = {
    "web_of_lies": {"sa_zs": (46.2, 18.8, 59.0, 31.6, 55.6, 29.1, 76.1, 31.6),
                    "sa_fs": (56.4, 35.9, 100.0, 95.7, 53.8, 28.2, 98.3, 97.4),
                    "a_fs": (51.7, 41.4, 96.6, 59.5, 56.9, 36.2, 100.0, 89.7)},
    "tracking_shuffled_objects_three_objects": {
        "sa_zs": (26.6, 2.1, 42.0, 11.9, 29.4, 9.1, 61.5, 24.5),
        "sa_fs": (26.6, 0.0, 61.5, 44.1, 27.3, 8.4, 59.4, 42.7),
        "a_fs": (16.8, 16.1, 76.6, 66.4, 34.3, 29.2, 69.3, 65.0)},
    "temporal_sequences": {"sa_zs": (55.6, 48.3, 56.5, 12.1, 61.6, 27.2, 69.0, 30.6),
                           "sa_fs": (81.0, 58.6, 81.0, 40.9, 67.2, 24.1, 94.8, 47.0),
                           "a_fs": (67.3, 58.0, 62.0, 16.7, 65.3, 67.3, 97.3, 96.0)},
    "sports_understanding": {"sa_zs": (64.2, 50.7, 75.7, 58.8, 64.2, 50.7, 62.2, 43.2),
                             "sa_fs": (73.6, 58.8, 93.9, 91.9, 76.6, 57.2, 92.4, 79.3),
                             "a_fs": (68.0, 61.3, 90.0, 49.3, 75.5, 60.5, 90.5, 89.8)},
    "snarks": {"sa_zs": (66.2, 46.8, 70.1, 14.3, 71.4, 37.7, 71.4, 27.3),
               "sa_fs": (68.8, 24.7, 64.9, 61.0, 74.3, 40.5, 73.0, 58.1),
               "a_fs": (50.7, 45.2, 74.0, 49.3, 85.5, 47.8, 81.2, 56.5)},
    "ruin_names": {"sa_zs": (69.5, 54.1, 51.5, 11.6, 77.6, 56.5, 72.0, 23.3),
                   "sa_fs": (67.4, 34.3, 62.7, 41.2, 87.7, 56.8, 89.1, 57.7),
                   "a_fs": (74.7, 72.7, 78.0, 76.7, 86.4, 86.4, 93.2, 91.7)},
    "navigate": {"sa_zs": (65.5, 13.5, 61.5, 18.9, 48.8, 12.6, 63.8, 33.9),
                 "sa_fs": (56.1, 23.0, 91.9, 77.7, 56.8, 8.1, 88.5, 77.7),
                 "a_fs": (68.7, 51.3, 96.7, 35.3, 71.3, 47.3, 92.7, 88.0)},
    "movie_recommendation": {"sa_zs": (76.7, 68.1, 67.1, 18.6, 72.4, 53.8, 70.5, 49.5),
                             "sa_fs": (81.4, 52.9, 83.8, 32.9, 79.5, 70.5, 81.9, 65.7),
                             "a_fs": (88.0, 89.3, 93.3, 89.3, 84.9, 87.7, 90.4, 91.1)},
    "logical_deduction_five_objects": {"sa_zs": (41.7, 23.4, 44.0, 4.0, 44.8, 30.5, 55.7, 31.6),
                                       "sa_fs": (46.3, 10.9, 61.7, 38.9, 47.6, 28.2, 63.5, 46.5),
                                       "a_fs": (45.3, 51.3, 62.0, 64.7, 52.3, 52.3, 63.1, 65.1)},
    "hyperbaton": {"sa_zs": (58.1, 46.6, 59.5, 14.2, 54.7, 42.6, 62.8, 31.1),
                   "sa_fs": (52.0, 22.3, 69.6, 12.8, 57.1, 27.9, 81.0, 25.9),
                   "a_fs": (98.0, 38.7, 65.3, 50.0, 98.0, 89.1, 77.6, 61.9)},
    "disambiguation_qa": {"sa_zs": (63.1, 35.7, 63.7, 38.2, 66.9, 48.4, 46.5, 40.8),
                          "sa_fs": (63.7, 35.7, 69.4, 48.4, 69.4, 48.4, 74.5, 61.1),
                          "a_fs": (64.0, 61.3, 63.3, 64.7, 73.3, 73.3, 78.7, 76.0)},
    "date_understanding": {"sa_zs": (48.1, 45.6, 66.0, 52.7, 53.9, 34.9, 71.4, 42.3),
                           "sa_fs": (54.4, 46.5, 77.2, 71.8, 57.3, 51.0, 83.0, 74.3),
                           "a_fs": (67.3, 60.7, 83.3, 80.0, 58.7, 62.7, 88.7, 87.3)},
    "causal_judgment": {"sa_zs": (53.7, 25.6, 69.5, 2.4, 59.8, 35.4, 59.8, 34.1),
                        "sa_fs": (65.9, 11.0, 65.9, 26.8, 73.2, 30.5, 63.4, 47.6),
                        "a_fs": (62.7, 61.4, 59.0, 55.4, 73.5, 72.3, 77.1, 67.5)},
}

# Table 10. (bias_type, model, few_shot) -> (%unfaith NoCoT, %unfaith CoT, %expl NoCoT, %expl CoT)
TABLE10 = {
    ("suggested_answer", "gpt", False): (27.3, 49.9, 90.9, 91.3),
    ("suggested_answer", "gpt", True): (36.3, 28.6, 95.3, 86.6),
    ("suggested_answer", "claude", False): (38.1, 41.2, 77.0, 86.9),
    ("suggested_answer", "claude", True): (35.4, 23.9, 84.8, 84.4),
    ("ans_always_a", "gpt", True): (18.2, 21.8, 65.4, 76.5),
    ("ans_always_a", "claude", True): (27.8, 10.7, 45.5, 47.2),
}

# Table 12: failed (dropped) items.  task -> (SA Claude FS, SA Claude ZS, SA GPT FS, SA GPT ZS, A Claude FS, A GPT FS, N, #demos)
TABLE12 = {
    "causal_judgment": (0, 0, 0, 0, 0, 0, 160, 8),
    "date_understanding": (0, 0, 0, 0, 83, 83, 300, 15),
    "disambiguation_qa": (0, 0, 0, 0, 0, 0, 228, 9),
    "hyperbaton": (1, 0, 0, 0, 5, 0, 300, 7),
    "logical_deduction_five_objects": (7, 4, 0, 0, 20, 19, 220, 8),
    "movie_recommendation": (0, 0, 0, 0, 35, 30, 300, 15),
    "navigate": (0, 54, 0, 0, 0, 0, 300, 12),
    "ruin_names": (15, 1, 0, 0, 71, 37, 300, 10),
    "snarks": (10, 0, 0, 0, 5, 0, 151, 14),
    "sports_understanding": (3, 0, 0, 0, 5, 0, 300, 15),
    "temporal_sequences": (0, 0, 0, 0, 0, 0, 300, 9),
    "tracking_shuffled_objects_three_objects": (0, 0, 0, 0, 0, 0, 220, 11),
    "web_of_lies": (0, 0, 0, 0, 0, 0, 220, 10),
}

# Table 5 (BBQ). (debias_instruction, model, few_shot) -> (%unfaith NoCoT, CoT, %expl-by-bias NoCoT, CoT)
TABLE5 = {
    (False, "gpt", False): (22.1, 26.1, 61.0, 59.2), (False, "gpt", True): (17.0, 23.5, 60.2, 56.1),
    (False, "claude", False): (29.5, 25.8, 57.3, 54.5), (False, "claude", True): (22.8, 20.6, 68.6, 62.5),
    (True, "gpt", False): (20.5, 24.9, 59.7, 60.0), (True, "gpt", True): (15.6, 22.1, 60.7, 51.8),
    (True, "claude", False): (20.2, 22.5, 48.9, 45.4), (True, "claude", True): (26.0, 17.2, 51.8, 50.6),
}

# Headline sample sizes quoted in §3.1 footnote 4.
N_EVAL = 3299
N_INCONSISTENT = {"suggested_answer": 2111, "ans_always_a": 1928}

MODEL_KEY = {"text-davinci-003": "gpt", "claude-v1": "claude"}

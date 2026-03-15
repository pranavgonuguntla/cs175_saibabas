from rouge_score import rouge_scorer


def compute_rouge(references, hypotheses, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, hyp in zip(references, hypotheses):
        s = scorer.score(ref, hyp)
        for k in agg:
            agg[k].append(s[k].fmeasure)
    return {k: sum(v) / len(v) * 100 if v else 0.0 for k, v in agg.items()}

def cohen_kappa(ann1, ann2):
    
    # Compute Cohen's kappa for pair-wise annotators
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)   # Observed agreement

    uniq = set(ann1 + ann2)

    E = 0
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count   # Expected agreement

    return round((A - E) / (1 - E), 4) # Cohen's kappa calculation

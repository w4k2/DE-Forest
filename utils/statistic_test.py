import numpy as np
from scipy.stats import rankdata
from scipy import stats

# Calculate ranks for every method based on mean_scores; the higher the rank, the better the method
def calc_ranks(mean_scores, metric_id):
    ranks = []
    for ms in mean_scores[metric_id]:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    # print("\nRanks for", metric_a, ": ", ranks, "\n")
    mean_ranks = np.mean(ranks, axis=0)
    # print("\nMean ranks for", metric_a, ": ", mean_ranks, "\n")
    return(ranks, mean_ranks)

# Calculate Friedman statistics - implementation from Demsar2006
def friedman_test(metric_a, clf_names, mean_ranks, n_streams, critical_difference):
    N_ = n_streams
    k_ = len(clf_names)
    p_value = 0.05
    
    friedman = (12*N_/(k_*(k_+1)))*(np.sum(mean_ranks**2)-(k_*(k_+1)**2)/4)
    # print("Friedman", friedman)
    iman_davenport = ((N_-1)*friedman)/(N_*(k_-1)-friedman)
    # print("Iman-davenport", iman_davenport)
    f_dist = stats.f.ppf(1-p_value, k_-1, (k_-1)*(N_-1))
    # print("F-distribution", f_dist)
    if f_dist < iman_davenport:
        print("Reject hypothesis H0")
    
    # print("Critical difference", critical_difference)
    # print(mean_ranks)

    values = {
        "Metric": metric_a,
        "Friedman": friedman,
        "Iman-davenport": iman_davenport,
        "F-distribution": f_dist,
        "Critical_difference": critical_difference,
        "Mean_ranks": mean_ranks
    }

    return(values)

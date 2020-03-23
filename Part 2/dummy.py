import math 

def ndcg(relevance_scores):

    N = len(relevance_scores)
    if N == 0:
        return 0
    
    sorted_scores = sorted(relevance_scores, reverse=True)
    
    actual = relevance_scores[0]
    ideal = sorted_scores[0]
    for i in range(1, N):
        actual += relevance_scores[i]/math.log2(i + 1)
        ideal += sorted_scores[i]/math.log2(i + 1)
    
    return actual/ideal

if __name__ == "__main__":
    l = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    print(ndcg(l))
import sys
import csv
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
    
    if ideal == 0:
        return 0
    
    return actual/ideal

def ndcg_scores(relevance_scores):

    return ndcg(relevance_scores[:10]), ndcg(relevance_scores[:20])


if __name__ == "__main__":
    
    if (len(sys.argv) < 3):
        print("No of arguments mismatch\nUsage: python Assignment2_01_evaluator.py <path to​ gold_standard_ranked_list.csv> <path to Assignment2_01_ranked_list_<K>.csv>")

    actual_ranks_path = sys.argv[1]
    predicted_ranks_path = sys.argv[2]

    K = predicted_ranks_path[-5]

    # print(K)
    actual_results = {}
    with open(actual_ranks_path, 'r') as actual_file:
        reader = csv.reader(actual_file)
        actual_file.readline()
        for row in reader:
            query_id = row[0]
            if query_id in actual_results:
                actual_results[query_id][row[1]] = int(row[2])
            else:
                actual_results[query_id] = {}
                actual_results[query_id][row[1]] = int(row[2])

    predicted_results = {}
    with open(predicted_ranks_path, 'r') as predicted_file:
        reader = csv.reader(predicted_file)
        for row in reader:
            query_id = row[0]
            if query_id in predicted_results:
                predicted_results[query_id].append(row[1])
            else:
                predicted_results[query_id] = []
                predicted_results[query_id].append(row[1])
    
    results = []
    for query_id in predicted_results:
        
        # Consider top 20 ranked documents
        query_results = predicted_results[query_id][:20]
        # print("predicted", query_results)

        ap10 = 0
        relevant = 0
        retrieved = 0
        # print("actual", actual_results[query_id])
        for result in query_results[:10]:
            if result in actual_results[query_id]:
                relevant += 1
            retrieved += 1
            ap10 += relevant/retrieved

        ap20 = ap10
        ap10 /= 10

        for result in query_results[10:20]:
            if result in actual_results[query_id]:
                relevant += 1
            retrieved += 1
            ap20 += relevant/retrieved

        ap20 /= 20
        
        relevance_scores = []
        for result in query_results:
            if result in actual_results[query_id]:
                relevance_scores.append(actual_results[query_id][result])
            else:
                relevance_scores.append(0)
        
        ndcg10, ndcg20 = ndcg_scores(relevance_scores)

        results.append([query_id, ap10, ap20, ndcg10, ndcg20])

    file_name = "Assignment2_01_metrics_" + K +".csv"

    with open(file_name, 'w') as result_file:
        writer = csv.writer(result_file)
        writer.writerows(results)
    
    # Calculating averages
    avg_ap10 = 0
    avg_ap20 = 0
    avg_ndcg10 = 0
    avg_ndcg20 = 0
    for result in results:
        avg_ap10 += result[1]
        avg_ap20 += result[2]
        avg_ndcg10 += result[3]
        avg_ndcg20 += result[4]
    
    N = len(results)
    avg_ap10 /= N
    avg_ap20 /= N
    avg_ndcg10 /= N
    avg_ndcg20 /= N
    print("Mean Average Precision:")
    print("mAP@10 =", avg_ap10)
    print("mAP@20 =", avg_ap20)
    print("\nAverage NDCG:")
    print("averNDCG@10 =", avg_ndcg10)
    print("averNDCG@20 =", avg_ndcg20)  
    print("\nResults are stored in", file_name, "where each row indicates Query id, Average Precision@10, Average Precision@20, NDCG@10, NDCG@20 respectively.") 

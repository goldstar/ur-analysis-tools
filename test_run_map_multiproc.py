import predictionio
from tqdm import tqdm
import multiprocessing as mp
from config import init_config

from pyspark import SparkContext

from pyspark.sql import SQLContext

cfg = init_config('config.json')

def main():
    # cfg.testing.primary_event is purchased-event

    # with open(cfg.testing.non_zero_users_file, 'w') as output:
    #     output.write(','.join(non_zero_users))
    sc = SparkContext(cfg.spark.master, 'map_test: test')
    sqlContext = SQLContext(sc)
    test_df = sqlContext.read.json(cfg.splitting.test_file).select("entityId", "event", "targetEntityId").cache()

    test_data = test_df.filter("event = '%s'" % (cfg.testing.primary_event)).collect()

    _, r_data, _ = run_map_test(test_data, [cfg.testing.primary_event], test=False)
    non_zero_users = get_nonzero(r_data)
    # non_zero_users = get_nonzero(r_data)
    #
    # logging.info('Process "map separate events" test')
    #
    # logging.info("cfg.testing: %s", cfg.testing)
    # logging.info("cfg.testing.events: %s", cfg.testing.events)
    # for ev in cfg.testing.events:
    #     (r_scores, r_data, ipu) = run_map_test(test_data, [ev], users=non_zero_users, test=False)


def get_nonzero(r_data):
    users = [user for user, res_data in r_data.items() if res_data['scores'][0] != 0.0 or not cfg.testing.consider_non_zero_scores_only]
    return users



def multi_send_query(holdoutUsers, user_queries):
    res_user = None
    q = {
        "user": user_queries["user"],
        "eventNames": user_queries["eventNames"],
        "num": user_queries["num"]}

    try:
        res = engine_client.send_query(q)
        # Sort by score then by item name
        tuples = sorted([(r["score"], r["item"]) for r in res["itemScores"]], reverse=True)
        scores = [score for score, item in tuples]
        items = [item for score, item in tuples]
        res_user = {
            "items": items,
            "scores": scores  # list of floats?
        }
    except predictionio.NotFoundError:
        logging.debug("Error: " % predictionio.NotFoundError)
        logging.debug("Error with user: %s" % user)
        print("Error with user: %s" % user)
    except Exception as e:
        logging.debug("General exception: " % e)

    return res_user


def run_map_test(data, eventNames, users=None, primaryEvent=cfg.testing.primary_event,
                 consider_non_zero_scores=cfg.testing.consider_non_zero_scores_only,
                 num=200, K=cfg.testing.map_k, test=False, predictionio_url="http://0.0.0.0:8000"):

    N_TEST = 2000
    d = {}
    res_data = {}
    engine_client = predictionio.EngineClient(url=predictionio_url)

    for rec in data:
        if rec.event == primaryEvent:
            user = rec.entityId
            item = rec.targetEntityId
            if not users or user in users:
                d.setdefault(user, []).append(item)

    if test:
        holdoutUsers = [*d.keys()][1:N_TEST]
    else:
        holdoutUsers = [*d.keys()]


    prediction = []
    ground_truth = []
    user_items_cnt = 0.0
    users_cnt = 0

    q = []
    for user in tqdm(holdoutUsers[:10]):
        q.append({
            "user": user,
            "eventNames": eventNames,
            "num": num})
    p = mp.Pool(32)
    res_data = p.map(multi_send_query, holdoutUsers, q)
    print("res_data: ", res_data)

    # Consider only non-zero scores
    if consider_non_zero_scores:
        for user_event_num, user_scores in res_data:
            if not user_scores:
                continue
            prediction.append(user_scores['items'])
            ground_truth.append(d.get(q[user_event_num]['user'], []))
            user_items_cnt += len(d.get(q[user_event_num]['user'], []))
            users_cnt += 1
    else:
        for user_event_num, user_scores in res_data:
            if not user_scores:
                continue
            prediction.append(user_scores['items'])
            ground_truth.append(d.get(q[user_event_num]['user'], []))
            user_items_cnt += len(d.get(q[user_event_num]['user'], []))
            users_cnt += 1


    logging.debug("ground_truth after user added %s: " % ground_truth)
    logging.debug("event_group: %s" %eventNames)

    return ([metrics.mapk(ground_truth, prediction, k) for k in range(1, K + 1)],
            res_data, user_items_cnt / (users_cnt + 0.00001))

if __name__ == '__main__':
    main()

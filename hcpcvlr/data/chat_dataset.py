import os
import json
import random



def load_scienceqa_data(cfgs):
    problems = json.load(open(os.path.join(cfgs["data_dir"], 'problems.json')))
    pid_splits = json.load(open(os.path.join(cfgs["data_dir"], 'pid_splits.json')))

    qids = pid_splits['%s' % (cfgs["test_split"])]
    qids = qids[:cfgs["test_number"]] if cfgs["test_number"] > 0 else qids
    print(f"number of test problems: {len(qids)}\n")

    if cfgs["txt_only"]:
        txt_qids = []
        for qid in qids:
            if problems[qid]['image'] == None:
                txt_qids.append(qid) 
        print('Number of text-only questions: ', len(txt_qids))

    # pick up shot examples from the training set
    shot_qids = cfgs["shot_qids"]
    train_qids = pid_splits['train']
    if shot_qids == None:
        assert cfgs["shot_number"] >= 0 and cfgs["shot_number"] <= 32
        shot_qids = random.sample(train_qids, cfgs["shot_number"])  # random sample
    else:
        shot_qids = [str(qid) for qid in shot_qids]

    print("training question ids for prompting: ", shot_qids, "\n")

    return problems, txt_qids, shot_qids

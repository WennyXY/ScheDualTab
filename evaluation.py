import json
import argparse
import utils.evaluator as evaluator

def wiki_eval(args):
    pred_list, gt_list = [], []
    with open(args.results_path, 'r') as file:
        for line in file:
            instance = json.loads(line.strip())
            if 'response' in instance.keys():
                pred_list.append(str(instance['response']))
                gt_list.append(str(instance['answer']))
            else:
                print(instance)
                return
    
    print(len(pred_list), len(gt_list))
    exactmatch, _ = evaluator.eval_exactmatch(pred_list, gt_list)
    print(f"EM Score = {format(exactmatch, '.4f')};")

def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    # Add arguments here
    parser.add_argument('--results_path', type=str, default="WikiTableQuestions/", help='Path to the dataset')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    print(wiki_eval(args))
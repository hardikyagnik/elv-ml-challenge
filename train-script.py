from eluvio.models import LGSceneSeg
from eluvio.utils import Mode, get_splits, Shots_Dataset, Loader, save_model
from eluvio.utils.evaluate_sceneseg import *
from eluvio.utils.constants import *
import torch
import os, pickle, json


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

def train(max_epoch=MAX_EPOCH):
    train_dataloader = Loader.get_DataLoader(Mode.TRAIN)
    val_dataloader = Loader.get_DataLoader(Mode.VAL)

    lgss = LGSceneSeg()
    lgss.to(device)

    optimizer = torch.optim.Adam(lgss.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_func = torch.nn.CrossEntropyLoss(torch.Tensor([1,9]).to(device))

    print(f"max_epoch={max_epoch}")

    for epoch in range(1,max_epoch+1):
        print(f"epoch={epoch}")
        lgss.train()
        train_iter = 1
        for idx, (place, cast, action, audio, target) in enumerate(train_dataloader):
            place = place.to(device)
            cast = cast.to(device)
            action = action.to(device)
            audio = audio.to(device)
            target = target.view(-1).to(device)

            optimizer.zero_grad()
            output = lgss(place, cast, action, audio)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch={epoch} iteration={train_iter} loss={loss}")
            train_iter+=1
        
        # Validation
        val_loss = 0
        total_batch = 0
        print(f"Starting Validating for Epoch={epoch}")
        with torch.set_grad_enabled(False):
            for idx, (place, cast, action, audio, target) in enumerate(val_dataloader):
                place = place.to(device)
                cast = cast.to(device)
                action = action.to(device)
                audio = audio.to(device)
                target = target.view(-1).to(device)
                output = lgss(place, cast, action, audio)
                loss = loss_func(output, target)
                
                val_loss += loss
                total_batch += 1
        print(f"Validation Average Loss = {val_loss/total_batch}")

    return lgss

def evaluate():
    filenames = os.listdir(OUTPUT_PATH)
    print("# of IMDB IDs:", len(filenames))
    gt_dict = dict()
    pr_dict = dict()
    shot_to_end_frame_dict = dict()

    for fn in filenames:
        x = pickle.load(open(os.path.join(OUTPUT_PATH,fn), "rb"))

        gt_dict[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"]
        pr_dict[x["imdb_id"]] = x["scene_transition_boundary_prediction"]
        shot_to_end_frame_dict[x["imdb_id"]] = x["shot_end_frame"]

    scores = dict()

    scores["AP"], scores["mAP"], _ = calc_ap(gt_dict, pr_dict)
    scores["Miou"], _ = calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict)
    scores["Precision"], scores["Recall"], scores["F1"], *_ = calc_precision_recall(gt_dict, pr_dict)

    print("Scores:", json.dumps(scores, indent=4))

def generate_outputs(model):
    data_splits = get_splits()
    all_splits = data_splits[Mode.ALL.value]
    all_dataset = Shots_Dataset(all_splits, Mode.ALL.value)
    print(f"## GENERATING RESULTS")
    for f_idx, (path, windows) in enumerate(all_dataset.windows):
        print(f"Loading file number:{f_idx}")
        predictions = []
        with open(path,'rb') as fn:
            movie_dict:dict = pickle.load(fn)
            print(f"- Generating result for {movie_dict['imdb_id']}")
            with torch.set_grad_enabled(False):
                for idx, (place,cast,action,audio,target) in enumerate(all_dataset._get_single_iterator(movie_dict, windows)):
                    output = model(place, cast, action, audio)
                    output = torch.nn.functional.softmax(output, dim=1)
                    _, prediction = torch.max(output, 1)
                    predictions.append(prediction)

                    imdb_id = movie_dict['imdb_id']+'.pkl'
                    result = {}
                    result['imdb_id'] = movie_dict['imdb_id']
                    result['shot_end_frame'] = movie_dict['shot_end_frame']
                    result['scene_transition_boundary_ground_truth'] = movie_dict['scene_transition_boundary_ground_truth']
                    result['scene_transition_boundary_prediction'] = torch.Tensor(predictions)
                    output_path = os.path.join(OUTPUT_PATH, imdb_id)
                    with open(output_path, "wb") as fout:
                        pickle.dump(result, fout, protocol=pickle.HIGHEST_PROTOCOL)

def main():

    print(f"\nSTARTING TRAINING \n")
    model = train()

    print(f"\nSAVING MODEL\n")
    save_model(model)
    
    ## Set name of the model that you want to use for predictions or pass the recent trained model
    # name = "model_161430132138.pth"
    # model = LGSceneSeg()
    # model.load_state_dict(torch.load(os.path.join(BINARIES_PATH, "model_161430132138.pth")))
    
    generate_outputs(model)

    print(f"\n EVALUATING \n")
    evaluate()
    
if __name__ == '__main__':
    main()
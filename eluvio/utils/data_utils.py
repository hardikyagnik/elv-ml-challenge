import torch
import torch.utils.data as data
import os
import pickle

from .constants import *


class Shots_Dataset(data.Dataset):
    def __init__(self, movie_path_map:dict, dataset_type:Mode):
        self.dataset_type = dataset_type # train, test, val
        self.windows: list = self._get_windows(movie_path_map)
        self._len_list = self._get_cumulative_len_list()

    def __len__(self):
        return sum([len(indices) for _, indices in self.windows])

    def __getitem__(self, index):
        # return data for 1 window
        win_idx = self._get_win_idx(index)
        index_ = index
        if win_idx > 0:
            index_ = index - self._len_list[win_idx-1]
        path, indices = self.windows[win_idx]
        win_indices = indices[index_] # 1x10 -> one window
        places = torch.Tensor([])
        casts = torch.Tensor([])
        actions = torch.Tensor([])
        audios = torch.Tensor([])
        targets = torch.Tensor([])

        with open(path,'rb') as fn:
            movie_dict:dict = pickle.load(fn)
            places_np = movie_dict['place'].numpy()
            casts_np = movie_dict['cast'].numpy()
            actions_np = movie_dict['action'].numpy()
            audio_np = movie_dict['audio'].numpy()
            targets_np = movie_dict['scene_transition_boundary_ground_truth'].numpy()

            for shot_idx in win_indices[:-1]:
                try:
                    places = torch.cat(
                        (places,
                        torch.Tensor(
                            [places_np[shot_idx, :], places_np[shot_idx+1, :]]).view(1,2,-1)
                            ))
                except:
                    print("FAILED")
                    print(f"Index = {index} and index_ = {index_}")
                    print(f"win_idx = {win_idx}")
                    print(f"_len_list = {self._len_list}")
                    print(f"win_indices = {win_indices}")
                    print(f"shot_id = {shot_idx}")


                casts = torch.cat(
                    (casts,
                    torch.Tensor(
                        [casts_np[shot_idx, :], casts_np[shot_idx+1, :]]).view(1,2,-1)
                        ))
                actions = torch.cat(
                    (actions,
                    torch.Tensor(
                        [actions_np[shot_idx, :], actions_np[shot_idx+1, :]]).view(1,2,-1)
                    ))
                audios = torch.cat(
                    (audios,
                    torch.Tensor(
                        [audio_np[shot_idx, :], audio_np[shot_idx+1, :]]).view(1,2,-1)
                    ))
                targets = torch.cat((targets,torch.Tensor([targets_np[shot_idx]])))
        
        # print(f"Shapes= {places.shape}, {casts.shape}, {actions.shape}, {audios.shape}, {targets.shape}")
        return places, casts, actions, audios, targets.long()

    def _get_windows(self, movie_path_map: dict):
        windows = []
        for _, path in movie_path_map.items():
            with open(path,'rb') as fn:
                movie_dict:dict = pickle.load(fn)
                shot_length:int = movie_dict['cast'].shape[0]
                indices: [] = []
                for idx in range(0, shot_length-1, 9):
                    curr_list:list = []
                    if idx+10 < shot_length-1:
                        for i in range(idx, idx+10):
                            curr_list.append(i)
                    else:
                        # dif = (idx+10)-shot_length
                        # previous = indices[-1]
                        # for i in range(10-dif, 10):
                        #     curr_list.append(previous[i])
                        # for i in range(idx, shot_length):
                        #     curr_list.append(i)
                        dif = shot_length-10
                        for i in range(dif,dif+10):
                            curr_list.append(i)
                    indices.append(curr_list)
                windows.append((path, indices))
        return windows

    def _get_cumulative_len_list(self):
        len_list = []
        for path, indices in self.windows:
            if not len(len_list):
                len_list.append(len(indices))
            else:
                len_list.append(len_list[-1]+len(indices))
        return len_list

    def _get_win_idx(self, index:int):
        for i, idx in enumerate(self._len_list):
            if index < idx:
                return i
    
    def _get_single_iterator(self, movie_dict, windows):
        places_np = movie_dict['place'].numpy()
        casts_np = movie_dict['cast'].numpy()
        actions_np = movie_dict['action'].numpy()
        audio_np = movie_dict['audio'].numpy()
        targets_np = movie_dict['scene_transition_boundary_ground_truth'].numpy()

        flatten_indices = [ind for indices in windows for ind in indices]
        unique_indices = sorted((set(flatten_indices)))
        for shot_idx in range(len(unique_indices)-1):
            places = torch.Tensor([places_np[shot_idx, :], places_np[shot_idx+1, :]]).view(1,2,-1)
            casts = torch.Tensor([casts_np[shot_idx, :], casts_np[shot_idx+1, :]]).view(1,2,-1)
            actions = torch.Tensor([actions_np[shot_idx, :], actions_np[shot_idx+1, :]]).view(1,2,-1)
            audios = torch.Tensor([audio_np[shot_idx, :], audio_np[shot_idx+1, :]]).view(1,2,-1)
            targets = torch.Tensor([targets_np[shot_idx]])
            yield places.view(1,1,2,-1), casts.view(1,1,2,-1), actions.view(1,1,2,-1), audios.view(1,1,2,-1), targets

 
def get_splits():
    movies = os.listdir(DATA_DIR_PATH)
    splits = {}
    splits[Mode.TRAIN.value] = {movie:os.path.join(DATA_DIR_PATH, movie) for movie in movies[:58]}
    splits[Mode.TEST.value] = {movie:os.path.join(DATA_DIR_PATH, movie) for movie in movies[58:61]}
    splits[Mode.VAL.value] = {movie:os.path.join(DATA_DIR_PATH, movie) for movie in movies[61:64]}
    splits[Mode.ALL.value] = {movie:os.path.join(DATA_DIR_PATH, movie) for movie in movies}
    return splits

class Loader():
    data_splits = get_splits()

    @classmethod
    def get_DataLoader(cls, dataset_type:Mode):
        split = cls.data_splits[dataset_type.value]
        dataset = Shots_Dataset(split, dataset_type.value)
        return data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)

def save_model(model):
    import time
    unique = int(time.time()*100)
    model_path = os.path.join(BINARIES_PATH, f'model_{unique}.pth')
    torch.save(model.state_dict(), model_path)

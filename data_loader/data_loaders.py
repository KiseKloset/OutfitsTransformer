import json, pickle
import math, random
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class PolyvoreDataLoader(DataLoader):
    def __init__(
        self, 
        split: str,
        data_dir: str, 
        is_disjoint: bool, 
        categories: list,
        masked_ratio: float,
        batch_size: int = 16, 
        shuffle: bool = True, 
        num_workers: int = 1,
        outfit_cap: int = -1,
    ):
        self.categories = categories
        self.polyvore_dataset = PolyvoreDataset(data_dir, is_disjoint, split, categories, masked_ratio, outfit_cap)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': default_collate,
            'num_workers': num_workers
        }
        super().__init__(dataset=self.polyvore_dataset, **self.init_kwargs)


    def query_top_items(self, embeddings: torch.Tensor, device, top_k: int = 5):
        return self.polyvore_dataset.query_top_items(embeddings, device, top_k)


    def get_random_item(self):
        return self.polyvore_dataset.get_random_item()


'''
self.index_names:  list[names]
self.index_embeddings: list[embeddings]
self.index_metadatas: dict[index, metadatas], some names will not have metadata.
self.outfits: list[dict[outfits. item_indices]]
'''
class PolyvoreDataset(Dataset):
    def __init__(self, data_dir: str, is_disjoint: bool, split: str, categories: list, masked_ratio: float, outfit_cap: int):
        self.data_dir: Path = Path(data_dir).resolve()
        self.categories = categories
        self.masked_ratio = masked_ratio

        self.__load_index_names()
        self.__load_index_embeddings()
        self.__load_index_metadatas()        
        self.__load_index_categories()
        self.__load_outfits(is_disjoint, split, outfit_cap)


    def __load_index_names(self):
        with open(self.data_dir / f"polyvore_index_names.pkl", "rb") as f:
            self.index_names = np.array(pickle.load(f))


    def __load_index_embeddings(self):
        self.index_embeddings = (
            torch.load(self.data_dir / "polyvore_index_embeddings.pt", map_location="cpu")
            .type(torch.float32)
        )
        self.index_embeddings = F.normalize(self.index_embeddings, dim=-1)


    def __load_index_metadatas(self):
        self.index_metadatas: dict = {}

        with open(self.data_dir / "polyvore_item_metadata.json", "r") as f:
            data = json.load(f)

        for idx, name in enumerate(self.index_names):
            if name in data:
                category = data[name]["semantic_category"]
                if category in self.categories:
                    self.index_metadatas[idx] = { "category": category }


    def __load_index_categories(self):
        self.index_categories = { i: [] for i in self.categories }
        for idx, metadata in self.index_metadatas.items():
            category = metadata["category"]
            self.index_categories[category].append(idx)            


    def __load_outfits(self, is_disjoint: bool, split: str, outfit_cap: int):
        if is_disjoint:
            outfits_dir = "disjoint"
        else:
            outfits_dir = "nondisjoint"

        # Read outfits from file
        self.outfits: list = []
        with open(self.data_dir / outfits_dir / f"{split}.json", "r") as f:
            data = json.load(f)
        
        # Create an inversed dictionary for faster lookup
        temp_dict = dict((names, idx) for idx, names in enumerate(self.index_names) if idx in self.index_metadatas)

        # Traverse the data in file
        for outfit in data:
            set_id = outfit["set_id"]
            items = outfit["items"]

            valid_items = [item for item in items if item in temp_dict]
            if len(valid_items) > 1:
                self.outfits.append({ "set_id": set_id, "items": [temp_dict[i] for i in valid_items] })

                outfit_cap -= 1
                if outfit_cap == 0:
                    return


    def __getitem__(self, index):
        n = len(self.categories)
        item_indices = torch.full((n,), -1).int()
        embeddings = torch.zeros(n, self.index_embeddings.shape[-1])
        fake_embeddings = torch.zeros(n, self.index_embeddings.shape[-1])
        input_mask = torch.full((n,), False) 
        target_mask = torch.full((n,), False) 

        # Read embeddings
        outfit = self.outfits[index]
        for item_idx in outfit["items"]:
            embedding = self.index_embeddings[item_idx]

            category = self.index_metadatas[item_idx]["category"] 
            category_idx = self.categories.index(category)

            item_indices[category_idx] = item_idx
            embeddings[category_idx] = embedding
            fake_embeddings[category_idx] = embedding
            input_mask[category_idx] = True

        # Mask partial to create target list
        available_idx = [idx for idx, i in enumerate(input_mask) if i]
        random.shuffle(available_idx)

        n_item_masked = math.ceil(min(max(1.0, self.masked_ratio * len(available_idx)), float(len(available_idx) - 1)))

        for i in available_idx[:n_item_masked]:
            input_mask[i] = False
            target_mask[i] = True
            fake_embeddings[i] = self.__get_random_item_with_same_category(item_indices[i].item())

        return item_indices, embeddings, input_mask, target_mask, fake_embeddings


    def __len__(self):
        return len(self.outfits)


    def __get_random_item_with_same_category(self, idx):
        cat = self.index_metadatas[idx]["category"]
        same_items = self.index_categories[cat]
        out = random.choice(same_items)
        while out == idx:
            out = random.choice(same_items)
        return self.index_embeddings[out]
    

    def query_top_items(self, embeddings: torch.Tensor, device, top_k: int = 5):
        out = torch.empty(embeddings.shape[0], len(self.categories), top_k, dtype=torch.int32).to(device)
        
        for i, category in enumerate(self.categories):
            query_embeddings = embeddings[:, i, :].squeeze(1)

            indices = np.array(self.index_categories[category])
            dataset_embeddings = self.index_embeddings[indices].to(device)

            cos_similarity = query_embeddings @ dataset_embeddings.transpose(0, 1)
            sorted_indices = torch.topk(cos_similarity, top_k, dim=1, largest=True).indices

            indices = torch.from_numpy(indices).to(device)
            out[:, i, :] = indices[sorted_indices]

        return out

    
    def get_random_item(self):
        index = random.randrange(len(self.index_embeddings))
        return self.index_embeddings[index]



class PolyvoreBenchmarkDataLoader(DataLoader):
    def __init__(
        self, 
        data_dir: str, 
        is_disjoint: bool, 
        batch_size: int = 16, 
        shuffle: bool = True, 
        num_workers: int = 1,
        masked_ratio: float = -1
    ):
        self.polyvore_dataset = PolyvoreBenchmarkDataset(data_dir, is_disjoint, masked_ratio)
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': default_collate,
            'num_workers': num_workers
        }
        super().__init__(dataset=self.polyvore_dataset, **self.init_kwargs)


    def query_top_items(self, embeddings, indices, device, top_k: int = 5):
        return self.polyvore_dataset.query_top_items(embeddings, indices, device, top_k)



class PolyvoreBenchmarkDataset(Dataset):
    def __init__(self, data_dir: str, is_disjoint: bool, masked_ratio: float):
        self.data_dir: Path = Path(data_dir).resolve()
        self.categories = [
            "all-body",
            "bags",
            "tops",
            "outerwear",
            "hats",
            "bottoms",
            "scarves",
            "jewellery",
            "accessories",
            "shoes",
            "sunglasses"
        ]
        self.masked_ratio = masked_ratio
        self.__load_index_names()
        self.__load_index_embeddings()
        self.__load_index_metadatas()        
        self.__load_index_categories()
        self.__load_outfits(is_disjoint)


    def __load_index_names(self):
        with open(self.data_dir / f"polyvore_index_names.pkl", "rb") as f:
            self.index_names = np.array(pickle.load(f))


    def __load_index_embeddings(self):
        self.index_embeddings = (
            torch.load(self.data_dir / "polyvore_index_embeddings.pt", map_location="cpu")
            .type(torch.float32)
        )
        self.index_embeddings = F.normalize(self.index_embeddings, dim=-1)


    def __load_index_metadatas(self):
        self.index_metadatas: dict = {}

        with open(self.data_dir / "polyvore_item_metadata.json", "r") as f:
            data = json.load(f)

        for idx, name in enumerate(self.index_names):
            if name in data:
                high = data[name]["semantic_category"]
                fine = data[name]["category_id"]
                self.index_metadatas[idx] = { "high": high, "fine": fine }           


    def __load_index_categories(self):
        self.index_categories = {}
        for idx, metadata in self.index_metadatas.items():
            category = metadata["fine"]
            if category not in self.index_categories:
                self.index_categories[category] = []
            self.index_categories[category].append(idx)


    def __load_outfits(self, is_disjoint: bool):
        if is_disjoint:
            outfits_dir = "disjoint"
        else:
            outfits_dir = "nondisjoint"

        # Read outfits from file
        self.outfits: list = []
        with open(self.data_dir / f"ocir_{outfits_dir}_test.json", "r") as f:
            data = json.load(f)
        
        # Create an inversed dictionary for faster lookup
        temp_dict = dict((names, idx) for idx, names in enumerate(self.index_names) if idx in self.index_metadatas)

        valid_counter_1 = 0
        valid_counter_2 = 0
        invalid_counter = 0

        self.blank_counter = {}

        # Traverse the data in file
        for outfit in data:
            set_id = outfit["set_id"]
            question = [temp_dict[i] for i in outfit["question"]]
            blank = temp_dict[outfit["blank"]]

            question, _type = self.__validate(question, blank)

            if len(question) > 0:
                self.outfits.append({ "set_id": set_id, "question": question, "blank": blank })
                if _type == 0:
                    valid_counter_1 += 1
                else:
                    valid_counter_2 += 1

                if blank not in self.blank_counter:
                    self.blank_counter[blank] = 0
                self.blank_counter[blank] += 1
            
            else:
                invalid_counter += 1

        print("Valid outfits 1: ", valid_counter_1)
        print("Valid outfits 2: ", valid_counter_2)
        print("Invalid outfits: ", invalid_counter)


    def __validate(self, question, blank):
        question_out = []
        highs = [self.index_metadatas[blank]["high"]]
        _type = 0

        for item in question:
            high = self.index_metadatas[item]["high"]
            if high not in highs:
                highs.append(high)
                question_out.append(item)
            else:
                _type = 1

        return question_out, _type 


    def __getitem__(self, index):
        n = len(self.categories)
        item_indices = torch.full((n,), -1).int()
        embeddings = torch.zeros(n, self.index_embeddings.shape[-1])
        fake_embeddings = torch.zeros(n, self.index_embeddings.shape[-1])
        input_mask = torch.full((n,), False) 
        target_mask = torch.full((n,), False) 

        # Read embeddings
        outfit = self.outfits[index]
        for item_idx in outfit["question"]:
            embedding = self.index_embeddings[item_idx]

            category = self.index_metadatas[item_idx]["high"] 
            category_idx = self.categories.index(category)

            item_indices[category_idx] = item_idx
            embeddings[category_idx] = embedding
            fake_embeddings[category_idx] = embedding
            input_mask[category_idx] = True

        if self.masked_ratio < 0:
            item_idx = outfit["blank"]
            category = self.index_metadatas[item_idx]["high"] 
            category_idx = self.categories.index(category)

            item_indices[category_idx] = item_idx
            embeddings[category_idx] = embedding
            fake_embeddings[category_idx] = self.__get_random_item_with_same_category(item_idx)
            target_mask[category_idx] = True

        else:
            available_idx = [idx for idx, i in enumerate(input_mask) if i]
            random.shuffle(available_idx)

            n_item_masked = math.ceil(min(max(1.0, self.masked_ratio * len(available_idx)), float(len(available_idx) - 1)))

            for i in available_idx[:n_item_masked]:
                input_mask[i] = False
                target_mask[i] = True
                fake_embeddings[i] = self.__get_random_item_with_same_category(item_indices[i].item())

        return item_indices, embeddings, input_mask, target_mask, fake_embeddings


    def __get_random_item_with_same_category(self, idx):
        cat = self.index_metadatas[idx]["fine"]
        same_items = self.index_categories[cat]
        out = random.choice(same_items)
        while out == idx:
            out = random.choice(same_items)
        return self.index_embeddings[out]


    def __len__(self):
        return len(self.outfits)


    def query_top_items(self, embeddings, item_indices, device, top_k: int = 5):
        out = torch.full((embeddings.shape[0], len(self.categories), top_k), fill_value = -1, dtype=torch.int32).to(device)
        
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if item_indices[i, j].item() == -1:
                    continue

                query_embedding = embeddings[i, j].view(1, -1)
                target_cat = self.index_metadatas[item_indices[i, j].item()]["fine"]
                
                indices = np.array(self.index_categories[target_cat])
                dataset_embeddings = self.index_embeddings[indices].to(device)

                cos_similarity = query_embedding @ dataset_embeddings.transpose(0, 1)
                k = min(top_k, len(indices))
                sorted_indices = torch.topk(cos_similarity, k, dim=1, largest=True).indices

                indices = torch.from_numpy(indices).to(device)
                out[i, j, :k] = indices[sorted_indices]
        
        return out

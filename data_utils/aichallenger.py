import os
import json
import torch
from PIL import Image

single_task = {
    0: "apple_no disease_healthy",
    1: "apple_scab_general",
    2: "apple_scab_serious",
    3: "apple_frogeye_spot",
    4: "apple_cedar apple rust_general",
    5: "apple_cedar apple rust_serious",
    6: "cherry_no disease_healthy",
    7: "cherry_powdery mildew_general",
    8: "cherry_powdery mildew_serious",
    9: "corn_no disease_healthy",
    10: "corn_cercospora zeae maydis tehon and daniels_general",
    11: "corn_cercospora zeae maydis tehon and daniels_serious",
    12: "corn_puccinia polysora_general",
    13: "corn_puccinia polysora_serious",
    14: "corn_curvularia leaf spot fungus_general",
    15: "corn_curvularia leaf spot fungus_serious",
    16: "corn_maize dwarf mosaic virus_general",
    17: "grape_no disease_healthy",
    18: "grape_black rot fungus_general",
    19: "grape_black rot fungus_serious",
    20: "grape_black measles fungus_general",
    21: "grape_black measles fungus_serious",
    22: "grape_leaf blight fungus_general",
    23: "grape_leaf blight fungus_serious",
    24: "citrus_no disease_healthy",
    25: "citrus_greening june_general",
    26: "citrus_greening june_serious",
    27: "peach_no disease_healthy",
    28: "peach_bacterial spot_general",
    29: "peach_bacterial spot_serious",
    30: "pepper_no disease_healthy",
    31: "pepper_scab_general",
    32: "pepper_scab_serious",
    33: "potato_no disease_healthy",
    34: "potato_early blight fungus_general",
    35: "potato_early blight fungus_serious",
    36: "potato_late blight fungus_general",
    37: "potato_late blight fungus_serious",
    38: "strawberry_no disease_healthy",
    39: "strawberry_scorch_general",
    40: "strawberry_scorch_serious",
    41: "tomato_no disease_healthy",
    42: "tomato_powdery mildew_general",
    43: "tomato_powdery mildew_serious",
    44: "tomato_early blight fungus_general",
    45: "tomato_early blight fungus_serious",
    46: "tomato_late blight water mold_general",
    47: "tomato_late blight water mold_serious",
    48: "tomato_leaf mold fungus_general",
    49: "tomato_leaf mold fungus_serious",
    50: "tomato_target spot bacteria_general",
    51: "tomato_target spot bacteria_serious",
    52: "tomato_septoria leaf spot fungus_general",
    53: "tomato_septoria leaf spot fungus_serious",
    54: "tomato_spider mite damage_general",
    55: "tomato_spider mite damage_serious",
    56: "tomato_ylcv virus_general",
    57: "tomato_ylcv virus_serious",
    58: "tomato_tomv_general"
}

multi_task = {
    "species": {
        0: "apple",
        1: "cherry",
        2: "corn",
        3: "grape",
        4: "citrus",
        5: "peach",
        6: "pepper",
        7: "potato",
        8: "strawberry",
        9: "tomato",
    },
    "disease": {
        0: "no disease", 
        1: "bacterial spot", 
        2: "black measles fungus", 
        3: "black rot fungus",
        4: "cedar apple rust",
        5: "cercospora zeae maydis tehon and daniels", 
        6: "curvularia leaf spot fungus",
        7: "early blight fungus", 
        8: "frogeye",
        9: "greening june", 
        10: "late blight fungus", 
        11: "late blight water mold",
        12: "leaf blight fungus", 
        13: "leaf mold fungus", 
        14: "maize dwarf mosaic virus",
        15: "powdery mildew", 
        16: "puccinia polysora",
        17: "scab",
        18: "scorch", 
        19: "septoria leaf spot fungus",
        20: "spider mite damage", 
        21: "target spot bacteria",
        22: "tomv",
        23: "ylcv virus", 
    },
    "severity": {
        0: "healthy",
        1: "general",
        2: "serious",
    }
}

tomato_task = {
    "disease": {
        0: "no disease",
        1: "early blight fungus",
        2: "late blight water mold",
        3: "leaf mold fungus",
        4: "no disease",
        5: "powdery mildew",
        6: "septoria leaf spot fungus",
        7: "spider mite damage",
        8: "target spot bacteria",
        9: "tomv",
        10: "ylcv virus",
    },
    "severity": {
        0: "healthy",
        1: "general",
        2: "serious",
    }
}

class AIChallenger(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training_set",
        num_tasks: str = "1",
        shuffle=False,
        transform=None,
    ):
        self.root = root
        self.num_tasks = num_tasks
        
        self.img_split = split
        self.transform = transform

        self.data = self.__load_files()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get data
        img_file, species, disease_class, severity = self.data[idx]

        img = Image.open(img_file)
        
        if self.transform is not None:
            img = self.transform(img)        
        
        if self.num_tasks == "1":
            label = {v: k for k, v in single_task.items()}[f"{species}_{disease_class}_{severity}"]
            return img, label

        elif self.num_tasks == "3":
            first_label = {v: k for k, v in multi_task["species"].items()}[f"{species}"]
            second_label = {v: k for k, v in multi_task["disease"].items()}[f"{disease_class}"]
            third_label = {v: k for k, v in multi_task["severity"].items()}[f"{severity}"]
            return img, first_label, second_label, third_label
        
        elif self.num_tasks == "2_tomato":
            label = {v: k for k, v in tomato_task.items()}[f"{disease_class}"]
            
            return img, label
    
    
    def __load_files(self):
        data = []

        dir = os.path.join(self.root, self.img_split)
        
        with open(os.path.join(dir, f"{self.img_split}_annotations.json"), "r") as f:
            all_labels = json.load(f)
        
        img_to_class = {entry["image_id"]: entry["disease_class"] for entry in all_labels}
        img_files = os.listdir(os.path.join(dir, "images"))
            
        for img in img_files:
            if img not in img_to_class:
                continue
            if img.endswith((".png", ".jpg", ".JPG", ".PNG")):

                disease_class_id = img_to_class[img]
                specie, disease, severity = single_task[disease_class_id].split("_", maxsplit=2)
                
                if self.num_tasks == "2_tomato" and specie != "tomato":
                    continue
                
                data.extend([(os.path.join(dir, "images", img), specie.lower().strip(), disease.lower().strip(), severity.lower().strip())])

        return data
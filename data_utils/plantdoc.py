import os
import torch
from PIL import Image

single_task = {
    0: "grape_leaf black rot",
    1: "strawberry_leaf",
    2: "corn_leaf blight",
    3: "tomato_leaf late blight",
    4: "tomato_leaf",
    5: "apple_leaf",
    6: "apple_rust leaf",
    7: "soyabean_leaf",
    8: "bell_pepper_leaf",
    9: "squash_powdery mildew leaf",
    10: "potato_leaf late blight",
    11: "corn_rust leaf",
    12: "tomato_leaf yellow virus",
    13: "tomato_leaf mosaic virus",
    14: "tomato_leaf bacterial spot",
    15: "blueberry_leaf",
    16: "grape_leaf",
    17: "peach_leaf",
    18: "cherry_leaf",
    19: "potato_leaf early blight",
    20: "bell_pepper_leaf spot",
    21: "apple_scab leaf",
    22: "corn_gray leaf spot",
    23: "tomato_septoria leaf spot",
    24: "raspberry_leaf",
    25: "tomato_leaf early blight",
    26: "tomato_mold leaf",
}

multi_task = {
    "species": {
        0: "grape",
        1: "strawberry",
        2: "corn",
        3: "tomato",
        4: "apple",
        5: "soyabean",
        6: "bell_pepper",
        7: "squash",
        8: "potato",
        9: "blueberry",
        10: "peach",
        11: "cherry",
        12: "raspberry",
    },
    "disease": {
        0: "leaf", # healthy
        1: "leaf black rot",
        2: "leaf blight",
        3: "leaf late blight",
        4: "rust leaf",
        5: "powdery mildew leaf",
        6: "leaf yellow virus",
        7: "leaf mosaic virus",
        8: "leaf bacterial spot",
        9: "leaf early blight",
        10: "leaf spot",
        11: "scab leaf",
        12: "gray leaf spot",
        13: "septoria leaf spot",
        14: "mold leaf",
    }
}

tomato_task = {
    0: "leaf", # healthy
    1: "leaf late blight",
    2: "leaf yellow virus",
    3: "leaf mosaic virus",
    4: "leaf bacterial spot",
    5: "septoria leaf spot",
    6: "leaf early blight",
    7: "mold leaf",
}

class PlantDoc(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
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
        img_file, species, disease_class = self.data[idx]

        img = Image.open(img_file).convert("RGB")  

        if self.transform is not None:
            img = self.transform(img)        
        
        if self.num_tasks == "1":
            label = {v: k for k, v in single_task.items()}[f"{species}_{disease_class}"]
            return img, label
        elif self.num_tasks == "2":
            first_label = {v: k for k, v in multi_task["species"].items()}[f"{species}"]
            second_label = {v: k for k, v in multi_task["disease"].items()}[f"{disease_class}"]
            
            return img, first_label, second_label
        
        elif self.num_tasks == "tomato":
            label = {v: k for k, v in tomato_task.items()}[f"{disease_class}"]
            
            return img, label
    
    
    def __load_files(self):
        data = []

        dir = os.path.join(self.root, self.img_split)
        for item in os.listdir(dir):
            specie, disease = item.split(" ", 1)
            if disease == "Early blight leaf":
                disease = "leaf early blight"
                
            if self.num_tasks == "tomato" and specie != "Tomato":
                continue
            
            for img in os.listdir(os.path.join(dir, item)):
                if img.endswith((".png", ".jpg", ".JPG", ".PNG")):
                    data.extend([(os.path.join(dir, item, img), specie.lower().strip(), disease.lower().strip())])

        return data
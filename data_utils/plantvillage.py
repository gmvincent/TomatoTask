import os
import torch
from PIL import Image

single_task = {
    0: "Cherry_Healthy",
    1: "Cherry_Powdery Mildew",
    2: "Potato_Healthy",
    3: "Potato_Early Blight",
    4: "Potato_Late Blight",
    5: "Strawberry_Healthy",
    6: "Strawberry_Leaf Scorch",
    7: "Apple_Healthy",
    8: "Apple_Black Rot",
    9: "Apple_Cedar Apple Rust",
    10: "Apple_Apple Scab",
    11: "Corn (Maize)_Healthy",
    12: "Corn (Maize)_Northern Leaf Blight",
    13: "Corn (Maize)_Cercospora Leaf Spot",
    14: "Corn (Maize)_Common Rust ",
    15: "Grape_Healthy",
    16: "Grape_Black Rot",
    17: "Grape_Leaf Blight",
    18: "Grape_Esca (Black Measles)",
    19: "Bell Pepper_Healthy",
    20: "Bell Pepper_Bacterial Spot",
    21: "Peach_Healthy",
    22: "Peach_Bacterial Spot",
    23: "Tomato_Healthy",
    24: "Tomato_Early Blight",
    25: "Tomato_Late Blight",
    26: "Tomato_Bacterial Spot",
    27: "Tomato_Yellow Leaf Curl Virus",
    28: "Tomato_Septoria Leaf Spot",
}

multi_task = {
    "species": {
        0: "Cherry",
        1: "Potato",
        2: "Strawberry",
        3: "Apple",
        4: "Corn (Maize)",
        5: "Grape",
        6: "Bell Pepper",
        7: "Peach",
        8: "Tomato"
    },
    "disease": {
        0: "Healthy",
        1: "Powdery Mildew",
        2: "Early Blight",
        3: "Late Blight",
        4: "Leaf Scorch",
        5: "Black Rot",
        6: "Cedar Apple Rust",
        7: "Apple Scab",
        8: "Northern Leaf Blight",
        9: "Cercospora Leaf Spot",
        10: "Common Rust ",
        11: "Leaf Blight",
        12: "Esca (Black Measles)",
        13: "Bacterial Spot",
        14: "Yellow Leaf Curl Virus",
        15: "Septoria Leaf Spot"
    }
}

tomato_task = {
    0: 'Healthy',
    1: 'Early Blight',
    2: 'Bacterial Spot',
    3: 'Yellow Leaf Curl Virus',
    4: 'Septoria Leaf Spot',
    5: 'Late Blight'
}

class PlantVillage(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "Train",
        num_tasks: str = "1",
        shuffle=False,
        transform=None,
    ):
        self.root = root
        self.num_tasks = num_tasks
        self.species = [d for d in os.listdir(self.root) if not d.startswith(".")]
        
        self.img_split = split
        self.transform = transform

        self.data = self.__load_files()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get data
        img_file, species, disease_class = self.data[idx]

        img = Image.open(img_file)
        
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

        for specie in self.species:
            dir = os.path.join(self.root, specie, self.img_split)
            
            if self.num_tasks == "tomato" and specie != "Tomato":
                continue
            
            for disease in os.listdir(dir):
                
                data.extend([
                    (os.path.join(dir, disease, img), specie, disease)
                    for img in os.listdir(os.path.join(dir, disease))
                    if img.endswith((".png", ".jpg", ".JPG", ".PNG"))
                ])
                
        return data
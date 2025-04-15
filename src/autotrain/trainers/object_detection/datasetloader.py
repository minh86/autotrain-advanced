import os
import json
import random
from collections import defaultdict
from datasets import Dataset, Features, Sequence, Value, Image as HFImage, ClassLabel


class LocalCocoDatasetLoader:
    def __init__(
        self,
        base_dir: str,
        train_dir="train",
        val_dir=None,
        test_dir=None,
        val_ratio=0.1,
        seed=42
    ):
        self.base_dir = base_dir
        self.val_ratio = val_ratio
        self.seed = seed

        self.split_dirs = {
            "train": os.path.join(base_dir, train_dir) if train_dir else None,
            "validation": os.path.join(base_dir, val_dir) if val_dir else None,
            "test": os.path.join(base_dir, test_dir) if test_dir else None,
        }

        self._loaded_splits = {}
        self._category_names = None
        self._features = None
        self._should_split_train = not val_dir or not os.path.exists(self.split_dirs["validation"])

    def __getitem__(self, split):
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}")

        if split == "validation" and self._should_split_train:
            if "train" not in self._loaded_splits:
                self._loaded_splits["train"] = self._load_split(self.split_dirs["train"])
            self._split_train_val()

        elif split not in self._loaded_splits:
            path = self.split_dirs[split]
            if not path or not os.path.exists(path):
                raise ValueError(f"Split '{split}' is not available and cannot be generated.")
            self._loaded_splits[split] = self._load_split(path)

        return self._loaded_splits[split]

    def keys(self):
        return ["train", "validation", "test"]

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return ((k, self[k]) for k in self.keys())

    def _split_train_val(self):
        full_dataset = self._loaded_splits["train"].shuffle(seed=self.seed)
        val_size = int(len(full_dataset) * self.val_ratio)
        self._loaded_splits["validation"] = full_dataset.select(range(val_size))
        self._loaded_splits["train"] = full_dataset.select(range(val_size, len(full_dataset)))

    def _load_split(self, split_path):
        annotation_path = os.path.join(split_path, "_annotations.coco.json")
        with open(annotation_path, "r") as f:
            coco = json.load(f)

        if self._category_names is None:
            self._category_names = [cat["name"] for cat in sorted(coco["categories"], key=lambda x: x["id"])]
            self._features = self._build_features()

        id2img = {img["id"]: img for img in coco["images"]}
        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        records = []
        for image_id, anns in img_to_anns.items():
            img_info = id2img[image_id]
            image_path = os.path.join(split_path, img_info["file_name"])
            objects = [{
                "id": ann["id"],
                "bbox": ann["bbox"],
                "category": self._category_names[ann["category_id"]],
                "category_id": ann["category_id"],
                "area": int(ann["area"]),
                "iscrowd": ann.get("iscrowd", 0)
            } for ann in anns]

            records.append({
                "image_id": image_id,
                "image": image_path,
                "width": img_info["width"],
                "height": img_info["height"],
                "objects": objects
            })

        return Dataset.from_list(records, features=self._features)

    def _build_features(self):
        return Features({
            "image_id": Value("int64"),
            "image": HFImage(),
            "width": Value("int32"),
            "height": Value("int32"),
            "objects": Sequence({
                "id": Value("int64"),
                "area": Value("int64"),
                "bbox": Sequence(Value("float32"), length=4),
                "category": ClassLabel(names=self._category_names),
                "category_id": Value("int64"),
                "iscrowd": Value("int64")
            })
        })

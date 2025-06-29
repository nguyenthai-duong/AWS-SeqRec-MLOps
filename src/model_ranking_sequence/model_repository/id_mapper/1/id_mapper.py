import json


class IDMapper:
    def __init__(self):
        self.user_to_index = {}
        self.index_to_user = []
        self.item_to_index = {}
        self.index_to_item = []
        self.unknown_user_index = -1
        self.unknown_item_index = -1

    def fit(self, user_ids, item_ids):
        self.user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.index_to_user = list(user_ids)
        self.item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.index_to_item = list(item_ids)
        self.unknown_user_index = len(self.user_to_index)
        self.unknown_item_index = len(self.item_to_index)

    def get_user_index(self, user_id):
        return self.user_to_index.get(user_id, self.unknown_user_index)

    def get_item_index(self, item_id):
        return self.item_to_index.get(item_id, self.unknown_item_index)

    def get_user_id(self, index):
        if index < len(self.index_to_user):
            return self.index_to_user[index]
        return "unknown_user"

    def get_item_id(self, index):
        if index < len(self.index_to_item):
            return self.index_to_item[index]
        return "unknown_item"

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(
                {
                    "user_to_index": self.user_to_index,
                    "index_to_user": self.index_to_user,
                    "item_to_index": self.item_to_index,
                    "index_to_item": self.index_to_item,
                },
                f,
            )

    def load(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.user_to_index = data["user_to_index"]
            self.index_to_user = data["index_to_user"]
            self.item_to_index = data["item_to_index"]
            self.index_to_item = data["index_to_item"]
            self.unknown_user_index = len(self.user_to_index)
            self.unknown_item_index = len(self.item_to_index)
        return self


def map_indice(df, idm: IDMapper, user_col="user_id", item_col="parent_asin"):
    return df.assign(
        **{
            "user_indice": lambda df: df[user_col].apply(
                lambda user_id: idm.get_user_index(user_id)
            ),
            "item_indice": lambda df: df[item_col].apply(
                lambda item_id: idm.get_item_index(item_id)
            ),
        }
    )

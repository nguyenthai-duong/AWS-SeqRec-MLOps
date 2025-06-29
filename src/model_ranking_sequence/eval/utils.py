import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from id_mapper import IDMapper


def create_label_df(
    df,
    user_col="user_id",
    item_col="parent_asin",
    rating_col="rating",
    timestamp_col="timestamp",
):
    label_cols = [user_col, item_col, rating_col, "rating_rank"]
    label_df = (
        df.sort_values([timestamp_col], ascending=[False])
        .assign(
            rating_rank=lambda df: df.groupby(user_col)[rating_col].rank(
                method="first", ascending=False
            )
        )
        .sort_values(["rating_rank"], ascending=[True])[label_cols]
    )
    return label_df


def create_rec_df(df, idm: IDMapper, user_col="user_id", item_col="parent_asin"):
    return df.assign(
        rec_ranking=lambda df: (
            df.groupby("user_indice", as_index=False)["score"].rank(
                method="first", ascending=False
            )
        ),
        **{
            user_col: lambda df: df["user_indice"].apply(
                lambda user_indice: idm.get_user_id(user_indice)
            ),
            item_col: lambda df: df["recommendation"].apply(
                lambda item_indice: idm.get_item_id(item_indice)
            ),
        }
    )


def merge_recs_with_target(
    recs_df,
    label_df,
    k=10,
    user_col="user_id",
    item_col="parent_asin",
    rating_col="rating",
):
    return (
        recs_df.pipe(
            lambda df: pd.merge(
                df,
                label_df[[user_col, item_col, rating_col, "rating_rank"]],
                on=[user_col, item_col],
                how="outer",
            )
        )
        .assign(
            rating=lambda df: df[rating_col].fillna(0).astype(int),
            # Fill the recall with ranking = top_K + 1 so that the recall calculation is correct
            rec_ranking=lambda df: df["rec_ranking"].fillna(k + 1).astype(int),
        )
        .sort_values([user_col, "rec_ranking"])
    )

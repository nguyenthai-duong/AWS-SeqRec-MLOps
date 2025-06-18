from functools import partial

import numpy as np
import pandas as pd
from loguru import logger


class InteractionDataSampler:
    """
    Just randomly get X users will not guarantee that the output dataset would qualify the condition of richness.
    Instead we take an iterative approach where we gradually drop random users from the dataset while keeping an eye on the conditions and our sampling target.
    """

    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        sample_users: int = 1000,
        min_val_records: int = 1000,
        random_seed: int = 41,
        min_user_interactions: int = 5,
        min_item_interactions: int = 10,
        buffer_perc: float = 0.2,
        perc_users_removed_each_round: float = 0.01,
        debug: bool = False,
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.sample_users = sample_users
        self.min_val_records = min_val_records
        self.random_seed = random_seed
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.buffer_perc = buffer_perc
        self.perc_users_removed_each_round = perc_users_removed_each_round
        self.debug = debug
        self.min_val_records = min_val_records

    def remove_random_users(self, df, k=10):
        users = df[self.user_col].unique()
        np.random.seed(self.random_seed)
        to_remove_users = np.random.choice(users, size=k, replace=False)
        return df.loc[lambda df: ~df[self.user_col].isin(to_remove_users)]

    def get_unqualified(self, df, col: str, threshold: int):
        unqualified = df.groupby(col).size().loc[lambda s: s < threshold].index
        return unqualified

    def sample(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        keep_random_removing: bool = True,
    ):
        get_unqualified_users = partial(
            self.get_unqualified,
            col=self.user_col,
            threshold=self.min_user_interactions,
        )
        get_unqualified_items = partial(
            self.get_unqualified,
            col=self.item_col,
            threshold=self.min_item_interactions,
        )

        r = 1

        sample_df = train_df.copy()

        while keep_random_removing:

            keep_removing = True
            i = 1

            num_users_removed_each_round = int(
                self.perc_users_removed_each_round * sample_df[self.user_col].nunique()
            )
            if r > 1:
                print(
                    f"\n\nRandomly removing {num_users_removed_each_round} users - Round {r} started"
                )
                new_sample_df = self.remove_random_users(
                    sample_df, k=num_users_removed_each_round
                )
            else:
                new_sample_df = sample_df.copy()

            while keep_removing:
                if self.debug:
                    logger.info(f"Sampling round {i} started")
                keep_removing = False
                uu = get_unqualified_users(new_sample_df)
                if self.debug:
                    logger.info(f"{len(uu)=:,.0f}")
                if len(uu):
                    new_sample_df = new_sample_df.loc[
                        lambda df: ~df[self.user_col].isin(uu)
                    ]
                    if self.debug:
                        logger.info(f"After removing uu: {len(new_sample_df)=:,.0f}")
                    assert len(get_unqualified_users(new_sample_df)) == 0
                    keep_removing = True
                ui = get_unqualified_items(new_sample_df)
                if self.debug:
                    logger.info(f"{len(ui)=:,.0f}")
                if len(ui):
                    new_sample_df = new_sample_df.loc[
                        lambda df: ~df[self.item_col].isin(ui)
                    ]
                    if self.debug:
                        logger.info(f"After removing ui: {len(new_sample_df)=:,.0f}")
                    assert len(get_unqualified_items(new_sample_df)) == 0
                    keep_removing = True
                i += 1

            sample_users = sample_df[self.user_col].unique()
            sample_items = sample_df[self.item_col].unique()
            num_users = len(sample_users)
            logger.info(f"After randomly removing users - round {r}: {num_users=:,.0f}")
            if (
                num_users > self.sample_users * (1 + self.buffer_perc) or r == 1
            ):  # First round always overriding sample_df with new_sample_df to keep all qualified items and users
                logger.info(
                    f"Number of users {num_users:,.0f} are still greater than expected, keep removing..."
                )
                sample_df = new_sample_df.copy()
            else:
                logger.info(
                    f"Number of users {num_users:,.0f} are falling below expected threshold, stop and use `sample_df` as final output..."
                )
                keep_random_removing = False

            val_sample_df = val_df.loc[
                lambda df: (
                    df[self.user_col].isin(sample_users)
                    & df[self.item_col].isin(sample_items)
                )
            ]
            if (num_val_records := val_sample_df.shape[0]) < self.min_val_records:
                logger.info(
                    f"Number of val_df records {num_val_records:,.0f} are falling below expected threshold, stop and use `sample_df` as final output..."
                )
                keep_random_removing = False

            r += 1

        sample_users = sample_df[self.user_col].unique()
        sample_items = sample_df[self.item_col].unique()
        logger.info(f"{len(sample_users)=:,.0f} {len(sample_items)=:,.0f}")

        return sample_df, val_sample_df

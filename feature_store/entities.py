from feast import Entity, ValueType


def define_user_entity():
    """
    Defines the user entity for the Feast feature store.

    Returns:
        Entity: Feast Entity object representing a user with a string-based user ID.
    """
    return Entity(
        name="user_id",
        value_type=ValueType.STRING,
        description="Unique identifier for a user."
    )


def define_parent_asin_entity():
    """
    Defines the parent ASIN entity for the Feast feature store.

    Returns:
        Entity: Feast Entity object representing a parent ASIN with a string-based identifier.
    """
    return Entity(
        name="parent_asin",
        value_type=ValueType.STRING,
        description="Unique identifier for a parent ASIN."
    )


user = define_user_entity()
parent_asin = define_parent_asin_entity()
from feast import Entity, ValueType

user = Entity(name="user_id", value_type=ValueType.STRING, description="User ID")
parent_asin = Entity(name="parent_asin", value_type=ValueType.STRING, description="Parent ASIN")

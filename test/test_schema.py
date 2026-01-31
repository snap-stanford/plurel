import pytest
import networkx as nx
from pathlib import Path

from torch_frame import stype
from plurel.schema import SQLSchemaGraphBuilder


def test_tables_present(parsed_tables):
    assert set(parsed_tables.keys()) == {"users", "orders", "products", "order_items"}


def test_users_table(parsed_tables):
    users = parsed_tables["users"]

    assert set(users["columns"].keys()) == {"user_id", "status"}
    assert users["pkey_col"] == "user_id"
    assert users["fkey_col_to_pkey_table"] == {}


def test_orders_table(parsed_tables):
    orders = parsed_tables["orders"]

    assert set(orders["columns"].keys()) == {
        "order_id",
        "user_id",
        "amount",
        "order_type",
    }
    assert orders["pkey_col"] == "order_id"
    assert orders["fkey_col_to_pkey_table"] == {"user_id": "users"}


def test_products_table(parsed_tables):
    products = parsed_tables["products"]

    assert set(products["columns"].keys()) == set(["product_id", "price"])
    assert products["pkey_col"] == "product_id"
    assert products["fkey_col_to_pkey_table"] == {}


def test_order_items_table(parsed_tables):
    oi = parsed_tables["order_items"]

    assert set(oi["columns"].keys()) == set(["order_id", "product_id", "quantity"])
    assert oi["pkey_col"] is None

    assert oi["fkey_col_to_pkey_table"] == {
        "order_id": "orders",
        "product_id": "products",
    }


def test_enum_columns(parsed_tables):
    users = parsed_tables["users"]
    orders = parsed_tables["orders"]

    status_col = users["columns"]["status"]
    order_type_col = orders["columns"]["order_type"]

    assert status_col["_stype"] == stype.categorical
    assert set(status_col["categories"]) == {"active", "inactive", "banned"}

    assert order_type_col["_stype"] == stype.categorical
    assert set(order_type_col["categories"]) == {"online", "instore"}


def test_key_columns_stype_without_categories(parsed_tables):
    users = parsed_tables["users"]
    orders = parsed_tables["orders"]

    user_id = users["columns"]["user_id"]
    fk_user_id = orders["columns"]["user_id"]

    assert user_id["_stype"] == stype.categorical
    assert user_id["categories"] is None

    assert fk_user_id["_stype"] == stype.categorical
    assert fk_user_id["categories"] is None


def test_numerical_columns_have_no_categories(parsed_tables):
    orders = parsed_tables["orders"]
    products = parsed_tables["products"]

    amount = orders["columns"]["amount"]
    price = products["columns"]["price"]

    assert amount["_stype"] == stype.numerical
    assert amount["categories"] is None

    assert price["_stype"] == stype.numerical
    assert price["categories"] is None


def test_graph_nodes(schema_graph):
    assert set(schema_graph.nodes) == {"users", "orders", "products", "order_items"}


def test_graph_edges(schema_graph):
    """
    Expected edges (ref -> child):
      users   -> orders
      orders  -> order_items
      products-> order_items
    """
    expected_edges = {
        ("users", "orders"),
        ("orders", "order_items"),
        ("products", "order_items"),
    }

    assert set(schema_graph.edges) == expected_edges


def test_graph_is_directed(schema_graph):
    assert isinstance(schema_graph, nx.DiGraph)


def test_node_attributes_attached(schema_graph):
    users = schema_graph.nodes["users"]

    assert "columns" in users
    assert "pkey_col" in users
    assert "fkey_col_to_pkey_table" in users

    assert users["pkey_col"] == "user_id"


def test_enum_columns_present_in_graph(schema_graph):
    users = schema_graph.nodes["users"]
    orders = schema_graph.nodes["orders"]

    status = users["columns"]["status"]
    order_type = orders["columns"]["order_type"]

    assert status["categories"] == ["active", "inactive", "banned"]
    assert order_type["categories"] == ["online", "instore"]


def test_build_graph_without_loading():
    builder = SQLSchemaGraphBuilder("does_not_matter.sql")

    with pytest.raises(ValueError, match="Schema not loaded"):
        builder.build_graph()


def test_composite_primary_key_not_supported(tmp_path: Path):
    sql = """
    CREATE TABLE bad_table (
        a INT,
        b INT,
        PRIMARY KEY (a, b)
    );
    """
    path = tmp_path / "bad.sql"
    path.write_text(sql)

    builder = SQLSchemaGraphBuilder(str(path))

    with pytest.raises(ValueError, match="Composite primary keys not supported"):
        builder.load_schema()

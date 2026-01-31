from pathlib import Path

import pytest
from plurel.schema import SQLSchemaGraphBuilder


@pytest.fixture
def schema_sql(tmp_path: Path):
    sql = """
    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY,
        status TEXT CHECK (status IN ('active', 'inactive', 'banned'))
    );

    CREATE TABLE orders (
        order_id BIGINT PRIMARY KEY,
        user_id BIGINT,
        amount DOUBLE,
        order_type TEXT CHECK (order_type IN ('online', 'instore')),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );

    CREATE TABLE products (
        product_id BIGINT PRIMARY KEY,
        price DOUBLE
    );

    CREATE TABLE order_items (
        order_id BIGINT,
        product_id BIGINT,
        quantity INT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    """
    path = tmp_path / "schema.sql"
    path.write_text(sql)
    return path


@pytest.fixture
def parsed_tables(schema_sql):
    builder = SQLSchemaGraphBuilder(str(schema_sql))
    builder.load_schema()
    return builder.tables


@pytest.fixture
def schema_graph(parsed_tables, schema_sql):
    builder = SQLSchemaGraphBuilder(str(schema_sql))
    builder.tables = parsed_tables
    return builder.build_graph()

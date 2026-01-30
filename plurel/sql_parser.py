import networkx as nx
from torch_frame import stype

from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import (
    Integer,
    BigInteger,
    Float,
    Numeric,
    Boolean,
    String,
    Date,
    DateTime,
    Time,
    Enum as SAEnum,
)


class SQLAlchemySchemaGraphBuilder:
    def __init__(self, sql_file: str):
        self.sql_file = sql_file
        self.metadata = MetaData()
        self.tables = {}

    def load_schema(self):
        sql_content = open(self.sql_file).read()
        engine = create_engine("sqlite:///:memory:")

        try:
            with engine.begin() as conn:
                for stmt in sql_content.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(text(stmt))
        except SQLAlchemyError as e:
            raise ValueError(f"Failed to execute SQL schema: {e}")

        self.metadata.reflect(bind=engine)

        for table_name, table in self.metadata.tables.items():
            self.tables[table_name] = self._build_table_ir(table)

    def _map_torch_frame_type(self, col):
        """
        Returns:
            {
                "_stype": stype,
                "categories": list[str] | None
            }
        """
        t = col.type

        enum_vals = self._get_enum_values(col)
        if enum_vals is not None:
            return {
                "_stype": stype.categorical,
                "categories": enum_vals,
            }
        if col.primary_key or col.foreign_keys:
            return {
                "_stype": stype.categorical,
                "categories": None,
            }
        if isinstance(t, Boolean):
            return {
                "_stype": stype.categorical,
                "categories": [0, 1],
            }
        if isinstance(t, (Float, Numeric, Integer, BigInteger)):
            return {
                "_stype": stype.numerical,
                "categories": None,
            }

        if isinstance(t, (Date, DateTime, Time, String)):
            raise ValueError(f"column: {col.name} of type: {t} is not supported")

    def _get_enum_values(self, col):
        t = col.type

        if isinstance(t, SAEnum):
            return list(t.enums)

        # --- SQLite-style CHECK constraint: col IN (...) ---
        for constraint in col.table.constraints:
            if not hasattr(constraint, "sqltext"):
                continue
            sql = str(constraint.sqltext).lower()
            name = col.name.lower()
            if f"{name} in" in sql:
                inside = sql.split("in", 1)[1]
                inside = inside.strip().lstrip("(").rstrip(")")
                vals = [v.strip().strip("'\"") for v in inside.split(",")]
                return vals

        return None

    def _build_table_ir(self, table):
        """
        Returns:
        {
            "columns":dict[col_name -> {
                "_stype": stype,
                "categories": list[str] | None
            }]
            "pkey_col": str | None,
            "fkey_col_to_pkey_table": dict[str, str],
        }

        Example:
        ```py
        {
            "columns": {
                "status": {
                    "_stype": stype.categorical,
                    "categories": ["open", "closed", "pending"]
                },
                "age": {
                    "_stype": stype.numerical,
                    "categories": None
                }
            },
            "pkey_col": "...",
            "fkey_col_to_pkey_table": {...},
        }
        ```
        """

        # ---------- Columns ----------
        columns = {}
        for c in table.columns:
            col_info = self._map_torch_frame_type(c)
            columns[c.name] = col_info

        # ---------- Primary key ----------
        pkeys = [c.name for c in table.primary_key.columns]
        if len(pkeys) > 1:
            raise ValueError(
                f"Composite primary keys not supported in simplified IR: {table.name} -> {pkeys}"
            )
        pkey_col = pkeys[0] if pkeys else None

        # ---------- Foreign keys ----------
        fkey_map = {}
        for fk in table.foreign_keys:
            local_col = fk.parent.name
            ref_table = fk.column.table.name
            fkey_map[local_col] = ref_table

        return {
            "columns": columns,
            "pkey_col": pkey_col,
            "fkey_col_to_pkey_table": fkey_map,
        }

    def build_graph(self) -> nx.DiGraph:
        """Build a NetworkX directed graph from schema."""
        if not self.tables:
            raise ValueError("Schema not loaded. Call load_schema() first.")

        G = nx.DiGraph()

        # Add nodes
        for table_name, table_info in self.tables.items():
            G.add_node(table_name, **{"name": table_name, **table_info})

        # Add edges: referenced_table -> current_table
        for table_name, table_info in self.tables.items():
            for _, ref_table in table_info["fkey_col_to_pkey_table"].items():
                G.add_edge(ref_table, table_name)

        return G

    def draw_graph(self, G: nx.DiGraph, filepath: str):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        plt.title("Database Schema Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filepath)

"""Execute the notebook in a fresh Python kernel and preserve outputs on failure."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / "prediction.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook, timeout=1800, kernel_name="python3",
    resources={"metadata": {"path": str(root)}}
)

def report_cell(cell, cell_index, **kwargs):
    print(f"Executing cell {cell_index + 1:02d}", flush=True)

client.on_cell_start = report_cell
try:
    client.execute()
finally:
    nbformat.write(notebook, path)

errors = [out for cell in notebook.cells if cell.cell_type == "code"
          for out in cell.get("outputs", []) if out.output_type == "error"]
assert not errors, errors
assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
print("All code cells executed successfully; notebook outputs saved.", flush=True)

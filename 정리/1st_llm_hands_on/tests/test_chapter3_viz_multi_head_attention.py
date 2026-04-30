import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("Chapter_3_Excercise_Viz_Multi_head_attention.ipynb")


class MultiHeadAttentionNotebookTests(unittest.TestCase):
    def test_interactive_attention_cell_updates_single_image_widget(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        interactive_cell = next(
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "def plot_attention_lines" in "".join(cell.get("source", []))
        )
        source = "".join(interactive_cell["source"])

        self.assertIn("import io", source)
        self.assertIn("plot_widget = widgets.Image(format=\"png\")", source)
        self.assertIn("buffer = io.BytesIO()", source)
        self.assertIn("fig.savefig(buffer, format=\"png\"", source)
        self.assertIn("plot_widget.value = buffer.getvalue()", source)
        self.assertNotIn("display(fig)", source)
        self.assertIn('previous_ui = globals().get("_attention_ui")', source)
        self.assertIn('previous_ui["plot_widget"].close()', source)
        self.assertIn('globals()["_attention_ui"] = {', source)


if __name__ == "__main__":
    unittest.main()

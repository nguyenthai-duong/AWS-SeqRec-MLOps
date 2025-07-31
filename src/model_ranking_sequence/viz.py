from typing import List

import plotly.graph_objects as go
import plotly.io as pio
from evidently.options import ColorOptions
from pydantic import BaseModel


class BlueQColors(BaseModel):
    main: str = "#114B8F"
    others: List[str] = [
        "#248F11",
        "#8F6211",
        "#8F116C",
        "#1A293A",
    ]  # Adobe Color Wheel > Square


blueq_colors = BlueQColors()

blueq_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter Variable, Inter"),
        title={
            "font": {
                "family": "Inter Variable, Inter Tight, Inter",
                "weight": "bold",
            }
        },
        colorway=[blueq_colors.main, *blueq_colors.others],
        template="simple_white",
        xaxis=dict(
            showgrid=False
        ),  # Do not put tickmode='linear' hear as it can cause Plotly to mistread xaxis as integer instead of category
        yaxis=dict(
            showgrid=False,
            # showticklabels=False,
            # tickformat=",",
        ),
        showlegend=False,
    )
)

pio.templates["blueq"] = blueq_template
pio.templates.default = "blueq"

color_scheme = ColorOptions(
    primary_color=blueq_colors.main,
    fill_color="#fff4f2",
    zero_line_color="#016795",
    current_data_color=blueq_colors.main,
    reference_data_color=blueq_colors.others[0],
)

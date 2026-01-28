import pandas as pd
from typing import Protocol, Any


TABLE_TEMPLATE = """
\\begin{{table}}[H]
    \centering
    \\begin{{tabular}}{{{alignement}}}
        \\toprule
        {header} \\\\
        \\midrule
        {rows} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Placeholder caption.}}
    \\label{{tab:placeholder_label}}
\\end{{table}}
"""

TABLE_EOL = " \\\\\n\t\t"
TABLE_SEP = " & "
TOP_RULE = "\\toprule"
MID_RULE = "\\midrule"
BOT_RULE = "\\bottomrule"


class AlignementProtocol(Protocol):
    def __call__(self, 
                 data_frame: pd.DataFrame,
                 ignore_index: bool = False,
                 **kwargs
                 ) -> str:
        pass


def default_alignement(data_frame: pd.DataFrame,
                       ignore_index: bool = False,
                       left_count: int | None = None,
                       **kwargs
                       ) -> str:
    """
        Returns an alignement for n(+1) columns, depending on ignore_index and left_count.
        When left_count is specified and ignore_index is true, one centered column is removed.

        :param data_frame: The pd.DataFrame to use as reference
        :param ignore_index: If true, removes one column from alignements.
        :param left_count: If specified, states the amount of columns aligned to the left.
        :param kwargs: Unused, for AlignementProtocol compatibility.
        :return: A string containing the LaTeX alignement
    """
    if left_count is None:
        if ignore_index:
            left_count = 0
        else:
            left_count = 1
        centered_count = len(data_frame.columns) - left_count
    else:
        centered_count = len(data_frame.columns) - left_count
        if not ignore_index:
            centered_count += 1

    alignements = ["l"] * left_count + ["c"] * centered_count
    alignement = "|" + "|".join(alignements) + "|"
    return alignement


def cell_to_latex(cell_data: Any) -> str:
    """
        Converts a cell into a latex string. Numeric values are encapsulated in `$`.

        :param cell_data: The data to convert.
        :return: A LaTeX string.
    """
    if (isinstance(cell_data, (int, float))
        or (isinstance(cell_data, str) and cell_data.isnumeric())
        ):
        cell_data = "${}$".format(cell_data)
        return cell_data
    
    return str(cell_data)


def row_data_to_latex(row_id: Any, 
                      row_data: pd.Series, 
                      ignore_index: bool = False,
                      ) -> str:
    """
        Converts a row into a latex string. Numeric values are encapsulated in `$`.

        :param row_id: The row identifier.
        :param row_data: The row data to convert.
        :param ignore_index: If true, row_id is not added to the string.
        :return: A LaTeX string.
    """

    if ignore_index:
        row = row_data.to_list()
    else:
        row = [row_id, *row_data.to_list()]
    
    row = [cell_to_latex(cell_data) for cell_data in row]
    row = TABLE_SEP.join(row)

    return row


def pandas_to_latex(data_frame: pd.DataFrame,
                    alignement_function: AlignementProtocol = default_alignement,
                    ignore_index: bool = False,
                    **kwargs
                    ) -> str:
    """
        Converts a pd.DataFrame into a LaTeX string using the `table` structure.
        Uses a simple layout with a single-line header separated from the rows by a \\midrule.
        Numeric values are converted to strings starting and ending with the math `$` sign.

        :param data_frame: The pd.DataFrame to convert.
        :param alignement_function: A function taking a pd.DataFrame and returning an alignement string.
        ignore_index

        :return: A LaTeX string.
    """
    alignement = alignement_function(data_frame, ignore_index, **kwargs)

    # region Header
    columns = data_frame.columns.to_list()
    if not ignore_index:
        columns.insert(0, data_frame.index.name)
    latex_header = TABLE_SEP.join(columns)
    # endregion

    # region Rows
    latex_rows: list[str] = []
    for row_id, row_data in data_frame.iterrows():
        latex_row = row_data_to_latex(row_id, row_data, ignore_index)
        latex_rows.append(latex_row)
    latex_rows_joint = TABLE_EOL.join(latex_rows)
    # endregion

    latex_table = TABLE_TEMPLATE.format(alignement=alignement, header=latex_header, rows=latex_rows_joint)
    return latex_table

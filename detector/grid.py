MAX_ROWS = 50
MAX_COLUMNS = 24
MAX_CELLS = 800


def allowed_shape(rows, columns):
    return (
        1 <= rows <= MAX_ROWS
        and 1 <= columns <= MAX_COLUMNS
        and rows * columns <= MAX_CELLS
    )

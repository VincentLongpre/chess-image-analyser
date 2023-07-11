import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import fen_to_labels
from PIL import Image
import os

# Path to the asset's PNG 1x with shadow images (download link in README)
chess_assets_path = "assets/chessSet/PNGs/With Shadow/1x"

labels_to_name_dict = {
    0: "empty",
    1: "b_pawn",
    2: "b_knight",
    3: "b_bishop",
    4: "b_rook",
    5: "b_queen",
    6: "b_king",
    7: "w_pawn",
    8: "w_knight",
    9: "w_bishop",
    10: "w_rook",
    11: "w_queen",
    12: "w_king",
}


def make_position(fen):
    """
    Function that takes a position's FEN notation and returns a PIL image object of the position.
    The image is constructed using the assets found here:

        https://opengameart.org/content/chess-pieces-and-board-squares

    Args:
        fen (str): The position's FEN notation
    """

    labels = fen_to_labels(fen)
    res = Image.new("RGB", (451 * 8, 451 * 8))

    for i in range(8):
        row = Image.new("RGB", (451 * 8, 451))

        for j in range(8):
            label = int(labels[i, j].item())
            name = labels_to_name_dict[label]

            if (i + j) % 2 == 1:
                square = Image.open(
                    os.path.join(chess_assets_path, "square brown dark_1x.png")
                )

            else:
                square = Image.open(
                    os.path.join(chess_assets_path, "square brown light_1x.png")
                )

            if name != "empty":
                piece = Image.open(os.path.join(chess_assets_path, f"{name}_1x.png"))
                x1 = int(0.5 * square.size[0] - 0.5 * piece.size[0]) + 1
                y1 = int(0.5 * square.size[1] - 0.5 * piece.size[1])
                square.paste(piece, box=(x1, y1), mask=piece)

            row.paste(square, (451 * j, 0))

        res.paste(row, (0, 451 * i))
    return res


if __name__ == "__main__":
    # Test the make_position function with a random position
    test_fen = "2b5-p2NBp1p-1bp1nPPr-3P4-2pRnr1P-1k1B1Ppp-1P1P1pQP-Rq1N3K"
    res = make_position(test_fen)
    plt.imshow(res)
    plt.savefig("test.png")

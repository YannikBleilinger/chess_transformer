from torch.utils.data import Dataset
import torch
import sqlite3
import chess
import json


class Chessset(Dataset):
    def __init__(self, db_path, split='train'):
        self.db_path = db_path
        self.split = split
        with sqlite3.connect(self.db_path) as temp_conn:
            cursor = temp_conn.cursor()
            self.len = cursor.execute("SELECT COUNT(*) FROM positions").fetchone()[0] - 1
            print(f"Len: {self.len}")
        self.train_len = int(self.len * 0.98)
        self.val_len = self.len - self.train_len

    def __len__(self):
        return self.train_len if self.split == 'train' else self.val_len

    def __getitem__(self, idx):
        if not self.split == 'train':
            idx = idx + self.train_len
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT fen, move_dist FROM positions WHERE rowid = ?", (idx + 1,))
        row = cur.fetchone()
        conn.close()

        if row is None:
            raise IndexError(f"Index {idx} out of range")
        fen, move_dist_json = row
        move_dist = json.loads(move_dist_json)
        move_tensor = self._move_dist_to_tensor(move_dist)
        board_tensor = self._fen_to_matrix(fen).float()

        return {"tokens": board_tensor, "targets": move_tensor, "fen": fen}

    def _fen_to_matrix(self, fen):
        board = chess.Board(fen)
        # 65 rows (64 squares + global features), 7 features per row
        matrix = torch.zeros(65, 7, dtype=torch.int8)

        # Global features (row 0)
        matrix[0, 0] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0  # White King Side
        matrix[0, 1] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0  # White King Side
        matrix[0, 2] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0  # White King Side
        matrix[0, 3] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0  # White King Side
        matrix[0, 4] = board.ep_square if board.ep_square else 0
        matrix[0, 5] = 1 if board.turn == chess.WHITE else -1
        matrix[0, 6] = board.halfmove_clock

        # Square features (rows 1-64)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                vec = torch.zeros(7, dtype=torch.long)
                vec[0] = 1 if piece.color == chess.WHITE else -1
                type_idx = {
                    chess.KING: 1,
                    chess.QUEEN: 2,
                    chess.ROOK: 3,
                    chess.BISHOP: 4,
                    chess.KNIGHT: 5,
                    chess.PAWN: 6
                }[piece.piece_type]
                vec[type_idx] = 1
                matrix[square + 1] = vec  # +1 to offset global row
        return matrix

    def _move_dist_to_tensor(self, move_dist):
        tensor = torch.full((4272,), float('-inf'))
        for key, value in move_dist.items():
            index = int(key)
            if 0 <= index < 4272:
                tensor[index] = value
            else:
                raise ValueError(f"Index {index} out of range in Chessset")

        return torch.nn.functional.softmax(tensor, dim=0)
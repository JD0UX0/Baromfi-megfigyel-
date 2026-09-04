from collections import deque
import time
import cv2


class RollingActivityGrid:

    def __init__(
        self,
        width,
        height,
        cell_size=140,
        window_seconds=60
    ):

        self.width = width
        self.height = height

        self.cell_size = cell_size
        self.window_seconds = window_seconds

        self.cols = width // cell_size + 1
        self.rows = height // cell_size + 1

        self.cells = [
            [
                deque()
                for _ in range(self.cols)
            ]
            for _ in range(self.rows)
        ]

    def update(self, points):

        now = time.time()

        for x, y in points:

            col = int(x // self.cell_size)
            row = int(y // self.cell_size)

            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.cells[row][col].append(now)

        cutoff = now - self.window_seconds

        for row in self.cells:
            for cell in row:

                while cell and cell[0] < cutoff:
                    cell.popleft()

    def render(self, frame):

        overlay = frame.copy()

        max_activity = 1

        for row in self.cells:
            for cell in row:
                max_activity = max(
                    max_activity,
                    len(cell)
                )

        for r in range(self.rows):
            for c in range(self.cols):

                activity = len(
                    self.cells[r][c]
                )

                intensity = activity / max_activity

                red = int(255 * intensity)
                green = int(255 * (1.0 - intensity))

                color = (
                    0,
                    green,
                    red
                )

                x1 = c * self.cell_size
                y1 = r * self.cell_size

                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    color,
                    -1
                )

        cv2.addWeighted(
            overlay,
            0.12,
            frame,
            0.72,
            0,
            frame
        )

        return frame
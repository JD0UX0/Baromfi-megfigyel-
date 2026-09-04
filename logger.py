import os
import csv
import datetime

class Logger:
    def __init__(self, dir_path='data'):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        self.current_date = self._today_string()
        self.file, self.writer = self._open_file_for_today()

    def _today_string(self):
        """YYYY-MM-DD dátum string."""
        return datetime.date.today().isoformat()

    def _open_file_for_today(self):
        """Megnyitja a napi CSV-t, létrehozza ha nem létezik, headerrel."""
        filename = f"{self.current_date} - Animals Anomaly Detector.csv"
        path = os.path.join(self.dir_path, filename)

        file_exists = os.path.isfile(path)

        f = open(path, "a", newline="", encoding="utf-8")
        writer = csv.writer(f)

        # csak akkor írunk headert, ha új fájl
        if not file_exists:
            writer.writerow([
                "date",
                "time",
                "video_path",
                "animal_count",
                "feeder_count",
                "drinker_count",
                "avg_move",
                "sleeping_count"
            ])
            f.flush()

        return f, writer

    def _check_date_rollover(self):
        """Ha átléptünk másnapra → új fájl."""
        today = self._today_string()
        if today != self.current_date:
            try:
                self.file.close()
            except:
                pass

            self.current_date = today
            self.file, self.writer = self._open_file_for_today()

    def log(self, video_path, total, feeder_count, drinker_count, avg_move, sleeping):
        """Egy sor log kiírása CSV-be."""
        self._check_date_rollover()

        now = datetime.datetime.now()
        date_str = now.strftime("%Y.%m.%d")
        time_str = now.strftime("%H:%M:%S")

        self.writer.writerow([
            date_str,
            time_str,
            video_path,
            total,
            feeder_count,
            drinker_count,
            avg_move,
            sleeping
        ])

        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except:
            pass

import os
import gspread


class SpreadsheetUpdater:
    SPREADSHEET_NAME = "PDB-DEV_ChatGPT"
    SCHEMA = [
        "DOI",
        "Title",
        "date of publishing",
        "date of analysis",
        "authors",
        "classification",
        "methods used",
        "software",
    ]

    def __init__(self):
        self.connect()
        self.spreadsheet = self.client.open(type(self).SPREADSHEET_NAME)
        self.worksheet = self.spreadsheet.get_worksheet(0)

    def connect(self):
        try:
            secret_file = os.path.join(os.getcwd(), "google_sheets_credentials.json")
            self.client = gspread.service_account(secret_file)
        except OSError as e:
            print(e)

    def append_row(self, row: [str]):
        """
        Adds a row to the spreadsheet, must follow SCHEMA:
        """
        self._check_row(row)
        self.worksheet.append_row(row)

    def append_rows(self, rows: [[str]]):
        """
        Adds a list of rows to the spreadsheet, each row must follow SCHEMA:
        """
        for row in rows:
            self._check_row(row)
        self.worksheet.append_rows(rows)

    def notify_arthur(self, message: str):
        """
        Args:
            message (str): _description_
        """
        self.spreadsheet.share(
            "aozalevsky@gmail.com",
            perm_type="user",
            role="writer",
            notify=True,
            email_message=message,
        )

    @staticmethod
    def _check_row(row: [str]):
        if len(row) != len(SpreadsheetUpdater.SCHEMA):
            raise ValueError(
                f"Row must have {len(SpreadsheetUpdater.SCHEMA)} fields in the order specified\n{SpreadsheetUpdater.SCHEMA}"
            )


def main():
    spread = SpreadsheetUpdater()
    dummy_row = [
        "DOI",
        "Title",
        "date of publishing",
        "date of analysis",
        "authors",
        "classification",
        "methods used",
        "software",
    ]
    spread.append_row(dummy_row)
    spread.append_rows([dummy_row, dummy_row, dummy_row])
    # spread.notify_arthur("testing out the code")


if __name__ == "__main__":
    main()

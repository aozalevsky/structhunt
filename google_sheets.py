import os
import gspread


class SheetsApiClient:
    """interface for all functionality with google sheets
    enables connection, append, and notification 
    """
    
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
        self.client = self.connect()
        self.spreadsheet = self.client.open(type(self).SPREADSHEET_NAME)
        self.worksheet = self.spreadsheet.get_worksheet(0)

    @staticmethod
    def connect():
        """connects to Google Sheets API service using private key file
        """
        try:
            secret_file = os.path.join(os.getcwd(), "google_sheets_credentials.json")
            return gspread.service_account(secret_file)
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

    def email(self, message: str, email_addresses: [str]):
        """Shares the spreadsheet with arthur, along with the message in an email
        Args:
            message (str): message to be sent
            email_addresses ([str]): recipients of notification
        """
        for email_address in email_addresses:
            self.spreadsheet.share(
                email_address,
                perm_type="user",
                role="writer",
                notify=True,
                email_message=message,
            )

    @staticmethod
    def _check_row(row: []):
        """Checks row

        Args:
            row ([]): row of values to be added to worksheet

        Raises:
            ValueError: number of values in rows doesn't match schema
        """
        if len(row) != len(SheetsApiClient.SCHEMA):
            raise ValueError(
                f"Row must have {len(SheetsApiClient.SCHEMA)} fields in the order specified\n{SheetsApiClient.SCHEMA}"
            )


def main():
    # some test code which initializes the client, then appends rows to the worksheet, then pings arthur
    spread = SheetsApiClient()
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

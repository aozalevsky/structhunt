import gspread

class SpreadsheetUpdater():
        
    credentials = {
    "type": "service_account",
    "project_id": "durable-sky-396700",
    "private_key_id": "5735737884fa17f981beceb424001445e2476ae3",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDcN1SaQ3pgvu/Z\nYL4VHQSoztL5gUBtDFHvICFGLMBLwxj1SI9zWeI0uH4QvW32QkqTwnYO8cW2jCpZ\n7Lu2ZDdW5copVt3p7GCacYC++hYjH8Y13FSotE0yWpxh8qIQexzgTcHenrCr8nAd\nhkeHyNwwUpmjOASIqOtIHj7cGqp43jxMSuwh8fK94ef+Aemo5+7h+tXlHqwMIFap\nyjdE4TNdJ+mYp2nm17PUYiP0Y+WbYEOeeo29So9P/Ir1gMpH5Fyu3RcBI7jloFZd\nM8Hfrj17KKMnVOKItfnYeLlFaBDSfgYP17v9NUfzHVUSDEa2T67mSfUG63aIiYSm\nXrtAuignAgMBAAECggEACDHe6hnjIfQfazcLm8mHdNvnEFxCkExKRQ9f1AN/HGw9\nyR+47UTF0DE7yVYWed8gDon8Aef2JyoY7ioksILfzeuhld9vq3BqbK59aTeK2PL2\n80yOfsCtTSRmEWPWeBQjKcDhaAfLva2F7CaLeH59aY1WJLOSJ57xmOHXQP8uozsm\nm8dMs1PgEELl2B2zc+6JtHnWH2CAxZiA2b9yh+iZi3kiaJyLIW1bgx4U3suDnsFc\n+Igk+AYsIZ4UuPwFxlb+2mvYpiZd/Br0ASnBDQvgXDA4Xlu8wBeukfun8VZOviE4\nFjdxYkHMLeCsu15Xsc3E3UOt8wIXbr6b9Wi7mitY4QKBgQD7oMvTGl3SMMBynYHx\nYbbqW15UksGx6oPXBeUsqmCc64qBTiTZwLh2gY0TOELa0Evlf5kVtEVvMPviFynf\nvEvEc7ZV9rqkpgD1YRk2oi98wgAPG2/xU/asdSNblLVT/tK21/a7agDuyh6CfRO6\nfzQf4GYITKjtI34kjkEYa4Y6TwKBgQDgCtGNu2TpqKhle6v59EQzyB5aUEg4LdJn\n+YTGppohtbtbbW2N4nhoOi+ibLtN0dIDetfSdZtXe7CC13WSVS7T9QAOC/u6g0rj\nQstqktfUUyasIPYKdWe64rNtNJkIW+x+bgz2p8fOTGKwkTSFUFtPYwvajwIUv6Zc\n2/Vjtt82qQKBgBKLcTonsU5yZVyNGyyNBQwUm8kj376bCAhq2M8H54LpIRYSiki6\nGV4yghEujk7OFync041z8cIWHBo3ltB0cikSVhfTzUGhMmTjORZ7sYBCU/rJDOD+\nTSm8oFR5izubhjAPjpGVaGgw4TrAuRl/knne8eYesDx55ywOh+Gi2wulAoGBANdW\nrqnKtyi6mfjI0LhzpmYa78mgpnmQ2U5kjtEc6sKB2S38VLNuPIr5ejVkyvb2OCRu\nGyjHL2L7mOF51CCtTVAeiUn3DKHtdbpPxhKOR3Jl5aLGH5ZX2DbRlOHfD0PwjrPK\ndR1SkIJh+u1484E7hjgcnBUbJUXqGy3foNGRwKPZAoGAZ1Ig6vyIbZk9Lnh08COS\nOQ6JrTEdDCfr1i3CapHAW+rN6oHlM+S7PmzTFuxrWhGAHDWDOBczrPa+ohUAmLWa\niSJDC+bBJvj/L0jD4qIm39ifDCSyZfoAkshvpEPe010tw3IuO64pV9wowbwyu+wN\nieOoIE/RPaDtfFb2IZG7pGA=\n-----END PRIVATE KEY-----\n",
    "client_email": "csv-updater@durable-sky-396700.iam.gserviceaccount.com",
    "client_id": "116349894744257971396",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/csv-updater%40durable-sky-396700.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
    }

    SPREADSHEET_NAME = "PDB-DEV_ChatGPT"
    
    def __init__(self):         
        self.gc = gspread.service_account_from_dict(type(self).credentials)
        sh = self.gc.open(type(self).SPREADSHEET_NAME)
        self.worksheet = sh.get_worksheet(0)

    def append_row(self, row: [str]):
        '''
        Adds a row to the spreadsheet, must follow schema:
        ['DOI', 'Title', 'date of publishing', 'date of analysis', 'authors', 'classification', 'methods used', 'software']
        '''
        self._check_row(row)
        self.worksheet.append_row(row)
    
    def append_rows(self, rows: [[str]]):
        '''
        Adds a list of rows to the spreadsheet, each row must follow schema:
        ['DOI', 'Title', 'date of publishing', 'date of analysis', 'authors', 'classification', 'methods used', 'software']
        '''
        for row in rows:
            self._check_row(row)
        self.worksheet.append_rows(rows)
    
    @staticmethod
    def _check_row(row: [str]):
        if len(row) != 8:
            raise ValueError("Row must have 8 fields in the order specified")

    def notify_arthur(self, message: str):
        self.sh.share('aozalevsky@gmail.com', perm_type='user', role='writer', notify=True, email_message=message) 



def main():
    spread = SpreadsheetUpdater()
    dummy_row = ['DOI', 'Title', 'date of publishing', 'date of analysis', 'authors', 'classification', 'methods used', 'software']
    spread.append_row(dummy_row)
    spread.append_rows([dummy_row, dummy_row, dummy_row])
    spread.notify_arthur("testing out my dope code")

if __name__ == '__main__':
    main()
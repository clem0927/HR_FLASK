import oracledb

# oracledb.init_oracle_client()

DB_CONFIG = {
    "user": "hh",
    "password": "1234",
    "dsn": "localhost:1521/FREEPDB1"
}

def get_connection():
    """
    Oracle DB 연결 반환
    """
    return oracledb.connect(
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        dsn=DB_CONFIG["dsn"]
    )

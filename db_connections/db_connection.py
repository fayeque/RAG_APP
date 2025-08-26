import os
import oracledb

def get_oracle_connection():
    return oracledb.connect(
        user=os.environ["DB_USERNAME"],
        password=os.environ["DB_PASSWORD"],
        dsn=f"""(DESCRIPTION =
                   (ADDRESS = (PROTOCOL = TCP)(HOST = {os.environ['HOST']})(PORT = {os.environ['PORT']}))
                   (CONNECT_DATA =
                     (SERVER = DEDICATED)
                     (SERVICE_NAME = {os.environ['SERVICE_NAME']})
                   )
                 )"""
    )
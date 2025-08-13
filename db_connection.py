

import oracledb

def get_oracle_connection():
    return oracledb.connect(
        user="VEC23AI",
        password="VEC23AI",
        dsn="""(DESCRIPTION =
                  (ADDRESS = (PROTOCOL = TCP)(HOST = ofss-mum-5891.snbomprshared2.gbucdsint02bom.oraclevcn.com)(PORT = 1521))
                  (CONNECT_DATA =
                    (SERVER = DEDICATED)
                    (SERVICE_NAME = VEC23AIPDB)
                  )
                )"""
    )

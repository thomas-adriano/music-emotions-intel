import logging as log
from contextlib import contextmanager

import numpy as np
import psycopg2

from music_emotions_intel import utils, configurations


def init_db(initscript=configurations.DATABASE_INITIALIZATION_SCRIPT):
    with connect_db() as (conn, cursor):
        log.info("initializing database...")
        init_stmt = utils.read_file_contents(initscript).replace('\n', ' ')
        log.debug(init_stmt)
        cursor.execute(init_stmt)
        print('database successfully initialized!')


@contextmanager
def connect_db():
    try:
        conn = psycopg2.connect("dbname='musicintel' user='musicintel' host='localhost' password=''")
        cursor = conn.cursor()
        yield conn, cursor
    except (psycopg2.Error, TypeError) as e:
        log.error("an error has occurred while trying to interact with database:\n ---> %s" % e)
    finally:
        if conn and not conn.closed:
            conn.commit()
            conn.close()
        if cursor and not conn.closed:
            cursor.close()


def insert(tablename, valuestuples):
    with connect_db() as (conn, cursor):
        col_names = ', '.join([x for (x, y) in valuestuples])
        vals = [str(y) for (x, y) in valuestuples]
        col_vals = ', '.join(vals)
        placeholders = np.full((len(valuestuples)), "%s")
        values_placeholders = ", ".join(placeholders)
        stmt = "INSERT INTO %s (%s) VALUES (%s)" % (tablename, col_names, values_placeholders)
        log.debug("executing sql %s" % stmt)
        cursor.execute(stmt, tuple(vals))

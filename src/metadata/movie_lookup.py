"""D06 schema/key validation only. Metadata never enters graph normalization."""
import csv
from pathlib import Path

import duckdb

from src.graph.full_builder import file_sha256


def validate_movies(source: Path) -> dict:
    with source.open(encoding="utf-8-sig",newline="") as stream:
        if next(csv.reader(stream),None) != ["movieId","title","genres"]:
            raise ValueError("Expected movieId,title,genres header")
    with duckdb.connect() as con:
        con.execute("CREATE TEMP TABLE movies AS SELECT * FROM read_csv(?, header=true, auto_detect=false, columns={'movieId':'BIGINT','title':'VARCHAR','genres':'VARCHAR'})",[str(source)])
        row = con.execute("""SELECT count(*),count(*) FILTER (WHERE movieId IS NULL),
            count(movieId)-count(DISTINCT movieId), count(*) FILTER (WHERE movieId<=0),
            count(*) FILTER (WHERE title IS NULL OR trim(title)=''),
            count(*) FILTER (WHERE genres IS NULL OR trim(genres)=''),
            count(*) FILTER (WHERE genres='(no genres listed)') FROM movies""").fetchone()
    result = dict(zip(("rows","null_movie_ids","duplicate_key_rows","invalid_movie_ids",
                       "missing_titles","missing_genres","no_genres_listed"),row))
    if any(result[k] for k in ("null_movie_ids","duplicate_key_rows","invalid_movie_ids")):
        raise ValueError(f"Invalid movie metadata keys: {result}")
    result.update(source_sha256=file_sha256(source),status="PASS")
    return result

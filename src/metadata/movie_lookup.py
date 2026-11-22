"""D06 schema/key validation only. Metadata never enters graph normalization."""
import csv
from pathlib import Path

import duckdb

from src.artifacts import file_sha256


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


def load_movies(source: Path) -> tuple[dict[int, tuple[str | None, str | None]], dict]:
    """Validate keys before loading metadata; never let duplicates multiply Top-K rows."""
    profile = validate_movies(source)
    with duckdb.connect() as con:
        rows = con.execute("""SELECT movieId, title, genres FROM read_csv(?, header=true,
            auto_detect=false, columns={'movieId':'BIGINT','title':'VARCHAR','genres':'VARCHAR'})""",
            [str(source)]).fetchall()
    return {movie_id: (title, genres) for movie_id, title, genres in rows}, profile


def enrich_topk(raw_rows: list[dict], movies: dict) -> tuple[list[dict], dict]:
    """Order-preserving LEFT lookup by raw movie ID with explicit coverage counts."""
    enriched = []
    missing_id = missing_title = missing_genres = 0
    for row in raw_rows:
        metadata = movies.get(row["movie_raw_id"])
        found = metadata is not None
        if not found:
            missing_id += 1
        title, genres = metadata if found else (None, None)
        missing_title += title is None or not title.strip()
        missing_genres += genres is None or not genres.strip()
        enriched.append({**row, "title": title, "genres": genres, "metadata_found": found})
    return enriched, {"rows": len(enriched), "metadata_found": len(enriched) - missing_id,
                      "missing_movie_ids": missing_id, "missing_titles": missing_title,
                      "missing_genres": missing_genres}

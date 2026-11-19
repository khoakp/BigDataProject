import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.metadata.movie_lookup import validate_movies


class MovieValidationTest(unittest.TestCase):
    def test_valid_metadata_missing_values_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'movies.csv'
            path.write_text('movieId,title,genres\n1,"Toy, Movie",Action|Comedy\n2,,(no genres listed)\n3,X,\n',encoding='utf-8')
            report = validate_movies(path)
            self.assertEqual((report['rows'],report['missing_titles'],report['missing_genres'],report['no_genres_listed']),(3,1,1,1))

    def test_reject_bad_header_duplicate_and_null_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'movies.csv'
            for content in ('id,title,genres\n1,X,A\n','movieId,title,genres\n1,X,A\n1,Y,B\n',
                            'movieId,title,genres\n,X,A\n','movieId,title,genres\n0,X,A\n'):
                with self.subTest(content=content):
                    path.write_text(content,encoding='utf-8')
                    with self.assertRaises(ValueError):
                        validate_movies(path)

    def test_reject_noninteger_movie_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'movies.csv'
            path.write_text('movieId,title,genres\nnot-an-integer,X,Action\n',encoding='utf-8')
            with self.assertRaises(duckdb.Error):
                validate_movies(path)

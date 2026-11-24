import unittest
from datetime import datetime, timezone

import duckdb

from scripts.goal04_protocol_profile import summarize


class ProtocolProfileTest(unittest.TestCase):
    def test_boundary_duplicates_cold_omitted_seen_and_empty_month(self):
        ts = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())
        b, feb, april = map(ts, ('2020-01-01','2020-02-01','2020-04-01'))
        data = [(1,10,4,b-1),(2,20,5,b-1),(3,30,4,b-1),
                (1,10,5,b),(1,20,4,b),(1,20,5,b), # seen + repeated new pair
                (9,10,4,b),(3,10,4,b),(1,99,4,b),(1,30,4,b),
                (1,40,3,b),(1,20,5,feb),(2,10,4,april)]
        with duckdb.connect() as con:
            con.execute('CREATE TABLE raw(userId BIGINT,movieId BIGINT,rating DOUBLE,timestamp BIGINT)')
            con.executemany('INSERT INTO raw VALUES (?,?,?,?)', reversed(data))
            con.execute("CREATE TABLE nodes AS SELECT * FROM (VALUES ('user',1),('user',2),('movie',10),('movie',20)) t(node_type,raw_id)")
            rows = summarize(con,b)
        jan = rows[0]
        self.assertEqual(jan['rating_events'],8)
        self.assertEqual(jan['future_positive_pairs'],6)
        self.assertEqual(jan['within_window_repeat_events'],1)
        for name in ('cold_user','omitted_user','cold_movie','omitted_movie','seen','eligible'):
            self.assertEqual(jan[name+'_pairs'],1)
        self.assertEqual(jan['eligible_users'],1)
        self.assertEqual(jan['new_canonical_edges'],5)
        self.assertEqual(rows[1]['seen_pairs'],1)
        self.assertEqual(rows[2]['rating_events'],0)
        self.assertIsNone(rows[2]['pair_coverage'])
        self.assertTrue(rows[3]['partial'])
        self.assertEqual(rows[3]['end_exclusive_ts'],april+1)


if __name__ == '__main__':
    unittest.main()

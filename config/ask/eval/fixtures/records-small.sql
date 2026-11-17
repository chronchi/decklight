-- Ask eval fixture `records-small` (config/ask/README.md): the record index layout of decklight-record, 12 bot games.
-- Going first: 5 won, 2 lost (71.4%). Going second: 1 won, 3 lost, 1 undecided (25%). All: 6 of 11 decided (54.5%).
CREATE TABLE matches (id TEXT PRIMARY KEY, created TEXT, mode TEXT, deck0_name TEXT, deck0_hash TEXT, deck1_name TEXT,
  deck1_hash TEXT, opponent TEXT, result INTEGER, turns INTEGER, path TEXT, me INTEGER, first INTEGER, won INTEGER,
  my_deck TEXT, opp_deck TEXT, opp_type TEXT, kind TEXT);
INSERT INTO matches VALUES
('g00','2026-11-01T12:00:00','random','Blue-Eyes','h0','Branded','h1','bot',0,10,'',0,0,1,'Blue-Eyes','Branded','bot','duel'),
('g01','2026-11-02T12:00:00','random','Blue-Eyes','h0','Blue-Eyes','h1','bot',0,11,'',0,0,1,'Blue-Eyes','Blue-Eyes','bot','duel'),
('g02','2026-11-03T12:00:00','random','Blue-Eyes','h0','Branded','h1','bot',1,12,'',0,0,0,'Blue-Eyes','Branded','bot','duel'),
('g03','2026-11-04T12:00:00','random','Blue-Eyes','h0','Snake-Eye','h1','bot',0,13,'',0,0,1,'Blue-Eyes','Snake-Eye','bot','duel'),
('g04','2026-11-05T12:00:00','random','Blue-Eyes','h0','Branded','h1','bot',0,14,'',0,0,1,'Blue-Eyes','Branded','bot','duel'),
('g05','2026-11-06T12:00:00','random','Blue-Eyes','h0','Snake-Eye','h1','bot',1,15,'',0,0,0,'Blue-Eyes','Snake-Eye','bot','duel'),
('g06','2026-11-07T12:00:00','random','Blue-Eyes','h0','Blue-Eyes','h1','bot',0,16,'',0,0,1,'Blue-Eyes','Blue-Eyes','bot','duel'),
('g07','2026-11-08T12:00:00','random','Blue-Eyes','h0','Branded','h1','bot',0,17,'',0,1,1,'Blue-Eyes','Branded','bot','duel'),
('g08','2026-11-09T12:00:00','random','Blue-Eyes','h0','Snake-Eye','h1','bot',1,18,'',0,1,0,'Blue-Eyes','Snake-Eye','bot','duel'),
('g09','2026-11-10T12:00:00','random','Blue-Eyes','h0','Branded','h1','bot',1,19,'',0,1,0,'Blue-Eyes','Branded','bot','duel'),
('g10','2026-11-11T12:00:00','random','Blue-Eyes','h0','Blue-Eyes','h1','bot',1,20,'',0,1,0,'Blue-Eyes','Blue-Eyes','bot','duel'),
('g11','2026-11-12T12:00:00','random','Blue-Eyes','h0','Snake-Eye','h1','bot',-1,21,'',0,1,-1,'Blue-Eyes','Snake-Eye','bot','duel');

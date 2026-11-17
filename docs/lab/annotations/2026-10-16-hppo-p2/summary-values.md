# Annotation summary

- Inputs: runs/value-p2-20261015-2044/hppo-p2; games: 1488, 0 truncated, 50480 searched
- Commit 3ffc312d1514e94cfe91fee5b93fe726e855f1f9; 2026-10-16T15:12:39+00:00
- Evals are P(win | both sides continue like our policy P).

## Calibration (searcher view, non-truncated games)

| p-hat bin | n | searcher win rate | mean p-hat |
|---|---|---|---|
| 0.0-0.1 | 3279 | 0.068 | 0.048 |
| 0.1-0.2 | 2546 | 0.166 | 0.150 |
| 0.2-0.3 | 2739 | 0.252 | 0.251 |
| 0.3-0.4 | 3126 | 0.366 | 0.350 |
| 0.4-0.5 | 3515 | 0.531 | 0.449 |
| 0.5-0.6 | 3818 | 0.629 | 0.550 |
| 0.6-0.7 | 4130 | 0.736 | 0.651 |
| 0.7-0.8 | 4916 | 0.804 | 0.753 |
| 0.8-0.9 | 6481 | 0.886 | 0.853 |
| 0.9-1.0 | 15930 | 0.978 | 0.972 |

Brier by turn bucket: 1-2: 0.154, 3-4: 0.101, 5-8: 0.114, 9+: 0.129.
ECE: 0.0330.

## Winner sign accuracy

| bucket | search eval | card_adv | LP lead |
|---|---|---|---|
| <=2 | 0.77 (n=14173) | 0.63 (n=1087) | 0.43 (n=86) |
| 3 | 0.83 (n=9750) | 0.71 (n=1097) | 0.61 (n=748) |
| 4 | 0.88 (n=7342) | 0.74 (n=1051) | 0.69 (n=952) |
| 5+ | 0.83 (n=19215) | 0.74 (n=4800) | 0.67 (n=5035) |

## Mining: significant >= mistake per searched opportunity (top 20)

| kind | msg | turns | seat first | deck_s | hits/opps | rate |
|---|---|---|---|---|---|---|
| own | IDLECMD | 3-4 | True | SkyStrikerAce | 14/115 | 0.1217 |
| own | IDLECMD | 1-2 | False | SnakeEyeFire | 14/129 | 0.1085 |
| own | IDLECMD | 3-4 | True | SinfulSnake2 | 13/76 | 0.1711 |
| own | IDLECMD | 1-2 | False | SkyStrikerAce | 13/152 | 0.0855 |
| own | IDLECMD | 1-2 | False | SinfulSnake | 12/99 | 0.1212 |
| own | IDLECMD | 1-2 | False | SnakeEyeTear | 12/102 | 0.1176 |
| own | IDLECMD | 9+ | False | BestowedDragonF | 11/47 | 0.2340 |
| own | PLACE | 1-2 | False | SnakeEyeFire | 11/204 | 0.0539 |
| own | CARD | 1-2 | False | Branded60 | 10/100 | 0.1000 |
| own | IDLECMD | 1-2 | False | CyberDragon | 10/106 | 0.0943 |
| own | PLACE | 3-4 | True | SnakeEyeFire | 10/170 | 0.0588 |
| own | IDLECMD | 3-4 | True | SinfulSnakeKash | 9/107 | 0.0841 |
| own | PLACE | 1-2 | False | Floowandereeze2 | 9/109 | 0.0826 |
| own | IDLECMD | 1-2 | False | SnakeEyeAlter | 9/131 | 0.0687 |
| own | PLACE | 1-2 | False | BestowedDragon | 9/150 | 0.0600 |
| own | PLACE | 1-2 | False | Branded | 9/150 | 0.0600 |
| own | IDLECMD | 5-8 | False | SinfulSnake2 | 8/54 | 0.1481 |
| own | CARD | 1-2 | False | SnakeEyeTear | 8/74 | 0.1081 |
| own | IDLECMD | 1-2 | False | BestowedDragonF | 8/77 | 0.1039 |
| own | IDLECMD | 9+ | True | Shiranui | 8/110 | 0.0727 |

## Tempo (win-pp for the turn player)

| turn | seat | n | mean | p10 | p90 |
|---|---|---|---|---|---|
| 1 | 0 | 1488 | -2.0 | -19.7 | 17.7 |
| 2 | 1 | 1375 | 2.0 | -20.5 | 26.8 |
| 3 | 0 | 1181 | 0.7 | -19.2 | 23.1 |
| 4 | 1 | 956 | 0.6 | -17.6 | 21.1 |
| 5 | 0 | 742 | 1.4 | -13.9 | 19.5 |
| 6 | 1 | 560 | 1.3 | -15.7 | 23.5 |
| 7 | 0 | 456 | -1.2 | -16.4 | 16.2 |
| 8 | 1 | 368 | -0.6 | -16.9 | 17.2 |
| 9 | 0 | 302 | 0.5 | -15.0 | 19.4 |
| 10 | 1 | 239 | 0.6 | -16.7 | 19.7 |
| 11 | 0 | 201 | -0.0 | -10.7 | 13.7 |
| 12 | 1 | 167 | 0.6 | -12.0 | 16.6 |

Swing-turn share: T1: 0.27, T2: 0.28, T3: 0.16, T4+: 0.30.
Swing winner goes on to win: 0.823 (n=1488).

## Value calibration at turn starts (both seats, non-truncated games)

| turn bucket | n | ECE |
|---|---|---|
| <=2 | 5304 | 0.0670 |
| 3 | 2412 | 0.0231 |
| 4 | 1983 | 0.0196 |
| 5+ | 8163 | 0.0175 |

## Winner sign accuracy of v_comb

| bucket | v_comb |
|---|---|
| <=2 | 0.63 (n=2976) |
| 3 | 0.75 (n=1375) |
| 4 | 0.78 (n=1181) |
| 5+ | 0.80 (n=5364) |

## Disagreement |v_p0 + v_p1|

| bucket | n | mean |
|---|---|---|
| <=2 | 2328 | 0.372 |
| 3 | 1037 | 0.233 |
| 4 | 802 | 0.199 |
| 5+ | 2799 | 0.193 |

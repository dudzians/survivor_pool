python tournament_simulator.py --simulations 10 --target-date 3/20/2025 --target-player 662 --variance-factor 0
python tournament_simulator.py --simulations 10 --target-date 3/20/2025 --target-player 66 --variance-factor 0


ensure tiebreaking is working by asking how it's set up then running sims and ask it to point out any tiebreakers and explain why one player won over the other

remember that rounds 64/32 the favorites factor weighs towards seeds 4-6, in rounds 16/8 towards 2-4, and after that towards 1.  LOOKING AT THIS YEAR 3s AND 4S WERE HIGHEST PICKS IN FIRST ROUND SO LET'S CHANGE WEIGHTING TO BE 3S AND 4S THEN 5S, 6S, AND 2S THEN AFTER THAT 7S/8S/9S, THEN 10S/11S/1S....OR BETTER YET...PUT A FORMULA TOGETHER FOR EVERY ROUND!

in splashsports pool, day 7 and 8 are actually combined and teams have to make 2 picks total.  analyze after updating for this.

IN ORDER TO RUN THIS PROGRAM THE ORDER SHOULD BE:
1. GET TEAMS.CSV, SCHEDULE.CSV FILES SAVED DOWN
2. RUN SCRAPE_STANDINGS.PY WITH PROPER URL THEN RUN CONVERT_SURVIVOR_PICKS.PY TO GET SURVIVORPOO.CSV SAVED DOWN WHICH IS THEN CONVERTS/CREATES TEAMS.CSV
3. RUN THE ACTUAL TOURNAMENT_SIMULATOR.PY


when running on the big tourney my survivor_pool.csv ended up 2255 unique entries but per website it's 2265. 



please update seed boosting as follows:

round of 64 - boost 3&4 seeds the most, then 2&5 less so, then 6&7 less so, then 8&9 less so, then 10&11&12 less so, then 1, then 13, then fully negate 14&15&16 so they aren't chosen
round of 32 - boost 3&4 seeds the most, then 2&5 less so, then 6&7 less so, then 1 less so, then 10&11&12&13 less so, then 8&9, then fully negate 14&15&16 so they aren't chosen
round of 16 - boost 2&3 seeds the most, then 6&7 less so, then 1&8&9&10 less so, then 4&5 less so, then everybody else after that
round of 8 - boost 1 seeds the most then everybody else after that
round of 4 - boost 1 seeds the most then everybody else after that
round of 2 - boost 1 seeds the most then everybody else after that
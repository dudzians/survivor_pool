python tournament_simulator.py --simulations 1 --target-date 4/5/2025 --target-player 662 --favorites-factor 10 --variance-factor 2
python tournament_simulator.py --simulations 100 --target-date 3/27/2025 --target-player 66 --favorites-factor 10 --variance-factor 2
python tournament_simulator.py --simulations 100 --target-date 3/20/2025 --target-player 66 --favorites-factor 10 --variance-factor 2


ensure tiebreaking is working by asking how it's set up then running sims and ask it to point out any tiebreakers and explain why one player won over the other

remember that rounds 64/32 the favorites factor weighs towards seeds 4-6, in rounds 16/8 towards 2-4, and after that towards 1.  LOOKING AT THIS YEAR 3s AND 4S WERE HIGHEST PICKS IN FIRST ROUND SO LET'S CHANGE WEIGHTING TO BE 3S AND 4S THEN 5S, 6S, AND 2S THEN AFTER THAT 7S/8S/9S, THEN 10S/11S/1S....OR BETTER YET...PUT A FORMULA TOGETHER FOR EVERY ROUND!

in splashsports pool, day 7 and 8 are actually combined and teams have to make 2 picks total.  analyze after updating for this.

IN ORDER TO RUN THIS PROGRAM THE ORDER SHOULD BE:
1. GET TEAMS.CSV, SCHEDULE.CSV FILES SAVED DOWN
2. RUN SCRAPE_STANDINGS.PY WITH PROPER URL THEN RUN CONVERT_SURVIVOR_PICKS.PY TO GET SURVIVORPOO.CSV SAVED DOWN WHICH IS THEN CONVERTS/CREATES TEAMS.CSV
3. RUN THE ACTUAL TOURNAMENT_SIMULATOR.PY


when running on the big tourney my survivor_pool.csv ended up 2255 unique entries but per website it's 2265. 

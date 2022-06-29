import shot_transition_detection
import scrape_resulttables
import pandas as pd

# csv [team, ts_perf_start, ts_perf_end]
# where team is "last name female athlete / last name male athlete" and timestamps are "hh:mm:ss"
timestamps = pd.read_csv(r'0_resources/timestamps_skatingperformance.csv')

# import full Olympics replay
video_path = '0_resources/beijing2022_fullreplay.mp4'

# query link for competition results
results_url = 'https://skatingscores.com/q/event/?show_ranks=on&underline=&season_codes=2022&division_codes=sr&division_codes' \
              '=jr&event_codes=oly&discipline_codes=pairs&unit_country_codes=all&unit_name=%25&sort=score&limit=50&submit' \
              '=Submit '

# find individual video shots
scenes = shot_transition_detection.find_scenes(video_path, threshold=30.0)

# scrape results from competition
competition_results = scrape_resulttables.get_table(results_url)
competition_results.columns = competition_results.columns.str.replace(' ', '')

# only keep those video shots that show an actual skating performance
# add team names to individual scenes
# scenes_teams indexes every scene, holds its start and end times; and matches it with the team name
scenes_within_timestamps, scenes_teams = shot_transition_detection.get_scenes_within_timestamps(scenes, timestamps)

# add scores to individual scenes
scenes_annotated = scenes_teams.rename(columns={"team": "Team"}).merge(competition_results, on='Team', how="inner")
if len(scenes_annotated) != len(scenes_teams):
    print("Matching of scenes with scores resulted in loss of rows - Some rows couldn't be matched")

scenes_annotated.to_csv('scenes_annotated_01.csv')

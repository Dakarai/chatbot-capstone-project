import re

# create the data files
# replace robot_text with data scraped from reddit (future)
data_path_human = "human_text.txt"
data_path_robot = "robot_text.txt"

# create lines from the datafiles
with open(data_path_human, 'r', encoding="utf-8") as f:
    human_lines = f.read().split('\n')

with open(data_path_robot, 'r', encoding="utf-8") as f:
    robot_lines = f.read().split('\n')

# substitute all bracketed words with "hi" for the human text
human_lines = [re.sub(r"\[\w+\]", "hi", line) for line in human_lines]
# strip all non-words out and then concatenate into one string for each line
human_lines = [" ".join(re.findall(r"\w+", line)) for line in human_lines]

# again but for the robot lines
robot_lines = [re.sub(r"\[\w+\]", "", line) for line in robot_lines]
robot_lines = [" ".join(re.findall(r"\w+", line) for line in robot_lines)]


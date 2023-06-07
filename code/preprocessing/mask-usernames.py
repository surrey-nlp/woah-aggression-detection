import pandas as pd

df = pd.read_csv("/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Datasets/Raw/all-data-v0.csv", dtype = str, on_bad_lines='skip')

tweet = df["tweet"]
count = df["count"]
id = df["id"]

new_tweets = list()
for tweet in tweet:
  line = tweet.split(" ")
  new_line = ""
  for i in line:
    if len(i) != 0 and i[0] == '@':
      i = "@MASK"
    new_line += i + " "
  new_tweets.append(new_line)

new_df = pd.DataFrame(columns=['count', 'id', 'tweet'])
new_df["tweet"] = new_tweets
new_df["count"] = count
new_df["id"] = id

new_df.to_csv("all-data-v1.csv")

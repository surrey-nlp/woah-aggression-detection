## Install twint using this fork:
# !pip install git+https://github.com/woluxwolu/twint.git

## Find followers of a user
# c.Username = "NaziaNafis"
# c.Pandas = True
# twint.run.Followers(c)
# follow_df = twint.run.Followers(c)

## Find tweets of a user
# c.Search = "from:"NaziaNafis"
# c.Store_object = True
# c.Limit = 10
# twint.run.Search(c)
# tlist = c.search_tweet_list

import twint

c = twint.Config()

c.Search = ["demonetization"]
c.Lang = "en"
c.Limit = 5000
c.Geocode = 'India'
c.Store_csv = True

# c.Since = "2019-01-01"
# c.Until = "2019-02-01"

c.Filter_retweets = True 
c.Links = "exclude"   
c.Hide_output = True
c.Output = "topic_2.csv"

twint.run.Search(c)

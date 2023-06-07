## Install twint using this fork:
# !pip install git+https://github.com/woluxwolu/twint.git

import twint

c = twint.Config()

c.Search = ["demonetization"]   # Pass a list of keywords
c.Lang = "en"
c.Limit = 5000
c.Geocode = 'India'
c.Store_csv = True

# c.Since = "2019-01-01"    # Limit scraping to a time-frame
# c.Until = "2019-02-01"

c.Filter_retweets = True 
c.Links = "exclude"   
c.Hide_output = True
c.Output = "topic_2_demonetization.csv"

twint.run.Search(c)

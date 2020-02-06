
# Data Setup


```
from google.colab import drive
```


```
drive.mount('./gdrive')
```


```
!ls
```

    gdrive	sample_data



```
import os
```


```
os.chdir('./gdrive/My Drive/Google Colaboratory/Colab Notebooks/Data Science   Machine Learning/Beatles NLP/Data')
```


```
import pandas as pd
import numpy as np

import re
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

%matplotlib inline
```


```
!pip install tqdm --upgrade
```


```
from tqdm import tqdm, tqdm_notebook
```


```
# using tqdm to get a progress bar on long-loading processes
tqdm.pandas(tqdm_notebook)
```


```
# reading in the big Beatles data Excel file into a Pandas dataframe. 
# The Beatles data will continue everything we need, including song names, album names, songwriters, lyrics, and more

data = pd.read_excel('beatles_data.xlsx')
```

# Raw Data Exploration


```
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Lead Vocal(s)</th>
      <th>Year</th>
      <th>Lyrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>"12-Bar Original"</td>
      <td>Anthology 2</td>
      <td>John Lennon\n Paul McCartney\n George Harrison...</td>
      <td>Instrumental</td>
      <td>1965</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"Across the Universe"</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Words are flowing out like endless rain into a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>"All I've Got to Do"</td>
      <td>UK: With the Beatles\n US: Meet the Beatles!</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1963</td>
      <td>Whenever I want you around yeh \nAll I gotta d...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"All My Loving"</td>
      <td>UK: With the Beatles\n US: Meet the Beatles!</td>
      <td>McCartney</td>
      <td>McCartney</td>
      <td>1963</td>
      <td>Close your eyes and I'll kiss you\nTomorrow I'...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>"All Things Must Pass"</td>
      <td>Anthology 3</td>
      <td>Harrison</td>
      <td>Harrison</td>
      <td>1969</td>
      <td>Sunrise doesn't last all morning, \nA cloudbur...</td>
    </tr>
  </tbody>
</table>
</div>




```
list(data)
```




    ['Song', 'Album Debut', 'Songwriter(s)', 'Lead Vocal(s)', 'Year', 'Lyrics']




```
# We will remove any songs that have something for lyrics (aka non-instrumental songs)
data = data[data['Lyrics'].notnull()]
```


```
data['Songwriter(s)']
```




    1                         Lennon
    2                         Lennon
    3                      McCartney
    4                       Harrison
    5      McCartney\n (with Lennon)
                     ...            
    213                    McCartney
    214           Lennon\n McCartney
    215                    McCartney
    216                       Lennon
    217                       Lennon
    Name: Songwriter(s), Length: 213, dtype: object




```
data[data['Songwriter(s)'] == 'Lennon']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Lead Vocal(s)</th>
      <th>Year</th>
      <th>Lyrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>"Across the Universe"</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Words are flowing out like endless rain into a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>"All I've Got to Do"</td>
      <td>UK: With the Beatles\n US: Meet the Beatles!</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1963</td>
      <td>Whenever I want you around yeh \nAll I gotta d...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>"All You Need Is Love"</td>
      <td>Magical Mystery Tour</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1967</td>
      <td>Love, love, love\nLove, love, love\nLove, love...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>"Bad to Me"</td>
      <td>The Beatles Bootleg Recordings 1963</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1963</td>
      <td>If you ever leave me, I'll be sad and blue\nDo...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>"The Ballad of John and Yoko"</td>
      <td>UK: 1967–1970\n US: Hey Jude</td>
      <td>Lennon</td>
      <td>Lennon\n (with McCartney)</td>
      <td>1969</td>
      <td>Standing in the dock at Southampton\nTrying to...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>"Because"</td>
      <td>Abbey Road</td>
      <td>Lennon</td>
      <td>Lennon\n McCartney\n Harrison[26]</td>
      <td>1969</td>
      <td>Because the world is round it turns me on\nBec...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>"Child of Nature"</td>
      <td>Let It Be... Naked - Fly on the Wall bonus disc</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>On the road to Rishikesh\nI was dreaming more ...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>"Come Together"</td>
      <td>Abbey Road</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1969</td>
      <td>Here come old flat top\nHe come grooving up sl...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>"The Continuing Story of Bungalow Bill"</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Hey, Bungalow Bill\nWhat did you kill\nBungalo...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>"Cry Baby Cry”</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon\n (with McCartney)</td>
      <td>1968</td>
      <td>Cry baby cry\nMake your mother sigh\nShe's old...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>"Dear Prudence"</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Dear Prudence, won't you come out to play\nDea...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>"Dig a Pony"</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1969</td>
      <td>I dig a pony\nWell, you can celebrate anything...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>"Do You Want to Know a Secret?"</td>
      <td>UK: Please Please Me\n US: The Early Beatles</td>
      <td>Lennon</td>
      <td>Harrison</td>
      <td>1963</td>
      <td>You'll never know how much I really love you\n...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>"Don't Let Me Down"</td>
      <td>UK: 1967–1970\n US: Hey Jude</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1969</td>
      <td>Don't let me down, don't let me down\nDon't le...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>"Everybody's Got Something to Hide Except Me a...</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Come on come on, come on come on\nCome on its ...</td>
    </tr>
    <tr>
      <th>59</th>
      <td>"Girl"</td>
      <td>Rubber Soul</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1965</td>
      <td>Is there anybody going to listen to my story\n...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>"Glass Onion"</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>I told you about strawberry fields\nYou know t...</td>
    </tr>
    <tr>
      <th>63</th>
      <td>"Good Morning Good Morning"</td>
      <td>Sgt. Pepper's Lonely Hearts Club Band</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1967</td>
      <td>Nothing to do to save his life call his wife i...</td>
    </tr>
    <tr>
      <th>64</th>
      <td>"Good Night"</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Starr</td>
      <td>1968</td>
      <td>Now it's time to say good night\nGood night sl...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>"Happiness Is a Warm Gun"</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>She's not a girl who misses much\nDo do do do ...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>"Hello Little Girl"</td>
      <td>Anthology 1</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1962</td>
      <td>Hello little girl\nHello little girl\nHello li...</td>
    </tr>
    <tr>
      <th>76</th>
      <td>"Hey Bulldog"</td>
      <td>Yellow Submarine</td>
      <td>Lennon</td>
      <td>Lennon\n (with McCartney)</td>
      <td>1968</td>
      <td>Sheepdog, standing in the rain\nBullfrog, doin...</td>
    </tr>
    <tr>
      <th>80</th>
      <td>"I Am the Walrus"</td>
      <td>Magical Mystery Tour</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1967</td>
      <td>I am he as you are he as you are me\nAnd we ar...</td>
    </tr>
    <tr>
      <th>81</th>
      <td>"I Call Your Name"</td>
      <td>UK: "Long Tall Sally" EP\n US: The Beatles' Se...</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1964</td>
      <td>I call your name but you're not there\nWas I t...</td>
    </tr>
    <tr>
      <th>82</th>
      <td>"I Don't Want to Spoil the Party"</td>
      <td>UK: Beatles for Sale\n US: Beatles VI</td>
      <td>Lennon</td>
      <td>Lennon, with McCartney</td>
      <td>1964</td>
      <td>I don't want to spoil the party so I'll go,\nI...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>"I Feel Fine"</td>
      <td>UK: A Collection of Beatles Oldies\n US: Beatl...</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1964</td>
      <td>Baby's good to me, you know\nShe's happy as ca...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>"I Should Have Known Better"</td>
      <td>UK: A Hard Day's Night</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1964</td>
      <td>I should have known better with a girl like yo...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>"I Want You (She's So Heavy)"</td>
      <td>Abbey Road</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1969</td>
      <td>I want you\nI want you so bad\nI want you\nI w...</td>
    </tr>
    <tr>
      <th>94</th>
      <td>"If I Fell"</td>
      <td>UK: A Hard Day's Night\n US: Something New</td>
      <td>Lennon</td>
      <td>Lennon\n (with McCartney)</td>
      <td>1964</td>
      <td>If I fell in love with you\nWould you promise ...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>"I'll Be Back"</td>
      <td>UK: A Hard Day's Night\n US: Beatles '65</td>
      <td>Lennon</td>
      <td>Lennon and McCartney)</td>
      <td>1964</td>
      <td>You know if you break my heart I'll go\nBut I'...</td>
    </tr>
    <tr>
      <th>99</th>
      <td>"I'll Cry Instead"</td>
      <td>UK: A Hard Day's Night\n US: Something New</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1964</td>
      <td>I've got every reason on earth to be mad\n'Cau...</td>
    </tr>
    <tr>
      <th>102</th>
      <td>"I'm a Loser"</td>
      <td>UK: Beatles for Sale\n US: Beatles '65</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1964</td>
      <td>I'm a loser\nI'm a loser\nAnd I'm not what I a...</td>
    </tr>
    <tr>
      <th>105</th>
      <td>"I'm In Love"</td>
      <td>The Beatles Bootleg Recordings 1963</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1963</td>
      <td>I've got something to tell you, I'm in love,\n...</td>
    </tr>
    <tr>
      <th>107</th>
      <td>"I'm Only Sleeping"</td>
      <td>UK: Revolver\n US: Yesterday and Today</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1966</td>
      <td>When I wake up early in the morning\nLift my h...</td>
    </tr>
    <tr>
      <th>108</th>
      <td>"I'm So Tired"</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>I'm so tired, I haven't slept a wink\nI'm so t...</td>
    </tr>
    <tr>
      <th>114</th>
      <td>"It's Only Love"</td>
      <td>UK: Help!\n US: Rubber Soul</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1965</td>
      <td>I get high when I see you go by, (my oh my)\nW...</td>
    </tr>
    <tr>
      <th>119</th>
      <td>"Julia"</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Half of what I say is meaningless\nBut I say i...</td>
    </tr>
    <tr>
      <th>135</th>
      <td>"Mean Mr. Mustard"</td>
      <td>Abbey Road</td>
      <td>Lennon</td>
      <td>Lennon\n (with McCartney)</td>
      <td>1969</td>
      <td>Mean Mister Mustard sleeps in the park\nShaves...</td>
    </tr>
    <tr>
      <th>142</th>
      <td>"Not a Second Time"</td>
      <td>UK: With the Beatles\n US: Meet the Beatles!</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1963</td>
      <td>You know you made me cry,\nI see no use in won...</td>
    </tr>
    <tr>
      <th>148</th>
      <td>"One After 909"</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon\n (with McCartney)</td>
      <td>1969</td>
      <td>My baby said she's trav'ling on the one after ...</td>
    </tr>
    <tr>
      <th>154</th>
      <td>"Please Please Me"</td>
      <td>UK: Please Please Me\n US: The Early Beatles</td>
      <td>Lennon</td>
      <td>Lennon and McCartney</td>
      <td>1962</td>
      <td>Last night I said these words to my girl\nI kn...</td>
    </tr>
    <tr>
      <th>155</th>
      <td>"Polythene Pam"</td>
      <td>Abbey Road</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1969</td>
      <td>Well, you should see Polythene Pam\nShe's so g...</td>
    </tr>
    <tr>
      <th>157</th>
      <td>"Rain"</td>
      <td>UK: Rarities\n US: Hey Jude</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1966</td>
      <td>If the rain comes \nThey run and hide their he...</td>
    </tr>
    <tr>
      <th>158</th>
      <td>"Real Love"</td>
      <td>Anthology 2</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1980</td>
      <td>All my little plans and schemes\nLost like som...</td>
    </tr>
    <tr>
      <th>159</th>
      <td>"Revolution"</td>
      <td>UK: 1967-1970\n US: Hey Jude</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>You say you want a revolution\nWell, you know\...</td>
    </tr>
    <tr>
      <th>163</th>
      <td>"Sexy Sadie"</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Sexy Sadie, what have you done\nYou made a foo...</td>
    </tr>
    <tr>
      <th>168</th>
      <td>"She Said She Said"</td>
      <td>Revolver</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1966</td>
      <td>She said\nI know what it's like to be dead\nI ...</td>
    </tr>
    <tr>
      <th>173</th>
      <td>"Strawberry Fields Forever"</td>
      <td>Magical Mystery Tour</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1966</td>
      <td>Let me take you down\nCause I'm going to Straw...</td>
    </tr>
    <tr>
      <th>174</th>
      <td>"Sun King"</td>
      <td>Abbey Road</td>
      <td>Lennon</td>
      <td>Lennon\n (with McCartney and Harrison)</td>
      <td>1969</td>
      <td>Aaaaahhhhhhhhhh...\nHere comes the sun king\nH...</td>
    </tr>
    <tr>
      <th>179</th>
      <td>"Tell Me Why"</td>
      <td>UK: A Hard Day's Night\n US: Something New</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1964</td>
      <td>Tell me why you cried, and why you lied to me\...</td>
    </tr>
    <tr>
      <th>185</th>
      <td>"This Boy"</td>
      <td>UK: Rarities\n US: Meet the Beatles!</td>
      <td>Lennon</td>
      <td>Lennon\n (with McCartney and Harrison)</td>
      <td>1963</td>
      <td>That boy\nTook my love away\nHe'll regret it s...</td>
    </tr>
    <tr>
      <th>186</th>
      <td>"Ticket to Ride"</td>
      <td>Help!</td>
      <td>Lennon</td>
      <td>Lennon\n (with McCartney)</td>
      <td>1965</td>
      <td>I think I'm gonna be sad,\nI think it's today,...</td>
    </tr>
    <tr>
      <th>187</th>
      <td>"Tomorrow Never Knows"</td>
      <td>Revolver</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1966</td>
      <td>Turn off your mind, relax and float down strea...</td>
    </tr>
    <tr>
      <th>194</th>
      <td>"What's The New Mary Jane"</td>
      <td>Anthology 3</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>She looks as an African Queen\nShe eating twel...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>"When I Get Home"</td>
      <td>UK: A Hard Day's Night\n US: Something New</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1964</td>
      <td>Whoa, ah, woah, ah\nI got a whole lot of thing...</td>
    </tr>
    <tr>
      <th>205</th>
      <td>"Yer Blues"</td>
      <td>The Beatles</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Yes, I'm lonely\nWant to die\nYes, I'm lonely\...</td>
    </tr>
    <tr>
      <th>206</th>
      <td>"Yes It Is"</td>
      <td>UK: Rarities\n US: Beatles VI</td>
      <td>Lennon</td>
      <td>Lennon, McCartney and Harrison</td>
      <td>1965</td>
      <td>If you wear red tonight\nRemember what I said ...</td>
    </tr>
    <tr>
      <th>208</th>
      <td>"You Can't Do That"</td>
      <td>UK: A Hard Day's Night\n US: The Beatles Secon...</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1964</td>
      <td>I got something to say that might cause you pa...</td>
    </tr>
    <tr>
      <th>216</th>
      <td>"You're Going to Lose That Girl"</td>
      <td>Help!</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1965</td>
      <td>You're going to lose that girl\nYou're going t...</td>
    </tr>
    <tr>
      <th>217</th>
      <td>"You've Got to Hide Your Love Away"</td>
      <td>Help!</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1965</td>
      <td>Here I stand head in hand\nTurn my face to the...</td>
    </tr>
  </tbody>
</table>
</div>



# Pre-processing the Data

We will break down the data into individual lyric lines.


```
import spacy
import en_core_web_sm
```


```
def preprocessing_songs(df):

  df['Lyric Line'] = df['Lyrics'].apply(lambda x: x.split('\n'))
  df = df.set_index(['Song', 'Album Debut', 'Songwriter(s)', 'Lead Vocal(s)', 'Year'])['Lyric Line'].apply(pd.Series).stack().reset_index().drop('level_5',1)
  df.rename(columns={0:'Lyric Line'}, inplace=True)

  print('Removing quotes from song...')
  df['Song'] = df['Song'].apply(lambda x: remove_song_quotes(x))

  print('Cleaning songwriter(s) column...')
  df['Songwriter(s)'] = df['Songwriter(s)'].apply(lambda x: condense_band_members(x))
  
  print('Cleaning lead vocals(s) column...')
  df['Lead Vocal(s)'] = df['Lead Vocal(s)'].apply(lambda x: condense_band_members(x))

  print('Counting the number of distinct lines in the lyrics.')
  df['Number Lines'] = df['Lyric Line'].apply(lambda x: number_lines(x))

  print('Counting the number of words in each song.')
  df['Number Words'] = df['Lyric Line'].apply(lambda x: number_words(x))

  print('Calculating the average number of words in each line.')
  df['Average Words Per Line'] = df.apply(lambda row: words_per_line(row['Number Lines'], row['Number Words']), axis=1)

  print('Counting the number of apostrophes in each song.')
  df['Number Apostrophes'] = df['Lyric Line'].apply(lambda x: apostrophe_count(x))

  print('Calculating the average number of words in each line.')
  df['Average Words Per Apostrophe'] = df.apply(lambda row: words_per_apostrophe(row['Number Words'], row['Number Apostrophes']), axis=1)

  print('Cleaning lyrics...')
  df['Cleaned Lyrics'] = clean_text(df, 'Lyric Line')

  return df

  
```


```
def pre_processing_songs_nlp(df):
  print('Extracting NLP features...')
  df['NLP Features'] = df['Cleaned Lyrics'].progress_apply(lambda x: nlp_pipeline(x))

  df = df.dropna()

  df['Tokens'] = df['NLP Features'].apply(lambda x: x['tokens'])

  df['Lemmas'] = df['NLP Features'].apply(lambda x: x['lemmas'])

  df['Lemmas Text'] = df['Lemmas'].apply(lambda x: ' '.join(x))

  df['Album Debut'] = df['Album Debut'].str.replace(r"\n", " ")
  
  # Replace two spaces with one space
  df['Album Debut'] = df['Album Debut'].str.replace(r"  ", " ")

  df = df.reset_index(drop=True)

  return df

  
```


```
def nlp_features(df):
  # Create dataframe of syntactic dependency relation counts 'dep'
    df_dep = df['NLP Features'].progress_apply(lambda x: nlp_dict_to_df(x, 'dep_counter'))
    df_dep = df_dep.fillna(value = 0)

    # Create dataframe of coarse-grained parts-of-speech counts 'pos'
    df_pos = df['NLP Features'].progress_apply(lambda x: nlp_dict_to_df(x, 'pos_counter'))
    df_pos = df_pos.fillna(value = 0)

    # Create dataframe of stop word counts 'stop'
    df_stop = df['NLP Features'].progress_apply(lambda x: nlp_dict_to_df(x, 'stop_counter'))
    df_stop = df_stop.fillna(value = 0)

    # Create dataframe of fine-grained parts-of-speech counts 'tag'
    df_tag = df['NLP Features'].progress_apply(lambda x: nlp_dict_to_df(x, 'tag_counter'))
    df_tag = df_tag.fillna(value = 0)

    # Combine all NLP dataframes
    df_spacy = pd.concat([df_stop, df_pos, df_tag, df_dep], axis=1)

    df_spacy = df_spacy.reset_index(drop=True)

    return df_spacy
```


```
def remove_song_quotes(song_row):
  # Removes any quotations around songs
  try:
    song_row = re.sub(r'"', '', song_row)
  except:
    pass

  return song_row
```


```
def condense_band_members(row):
  # Create new pandas df column for either the individual songwriter or group songwriters
  if row =='Harrison':
    return row
  elif row == 'Lennon':
    return row
  elif row == 'McCartney':
    return row
  elif row == 'Starr':
    return row
  elif row == 'Instrumental':
    return row
  else:
    return 'Multiple Beatles'
```


```
def number_lines(lyrics_row):
  # Counts the number of distinct lines in the lyrics
  try:
    number_lines = lyrics_row.count('\n') + 1
    return number_lines
  except:
    return 0
```


```
def number_words(lyrics_row):
  # Counts the number of words in each song
  try:
    list_of_words = lyrics_row.split()
    number_words = len(list_of_words)
    return number_words
  except:
    return 0
```


```
def words_per_line(number_lines_row, number_words_row):
    # Calculates the average words per line
    
    if number_words_row == 0:
        # Don't want to return np.nan, return 0
        return 0
    else:
        average_words_per_line = number_words_row / number_lines_row
        return average_words_per_line
```


```
def apostrophe_count(lyrics_row):
    # Counts the number of apostrophes
    
    try:
        number_apostrophes = lyrics_row.count("'")
        return number_apostrophes
    except:
        return 0
```


```
def words_per_apostrophe(number_words_row, number_apostrophes_row):
    # Calculates the average number of words used per apostrophe used
    
    if number_words_row == 0 or number_apostrophes_row == 0:
        # Don't want to return np.nan, return 0
        return 0
    else:
        average_words_per_apostrophe = number_words_row / number_apostrophes_row
        return average_words_per_apostrophe
```


```
def clean_text(df, df_column):
    """
    Removing line breaks, special characters.
    Setting all letters to lowercase -- EXCEPT for pronoun I, kept capitalized.
    """
    
    # Replacing line breaks with one space
    df['Cleaned Lyrics'] = df[df_column].str.replace(r"\n", " ")
    
    # Removing special characters
    df['Cleaned Lyrics'] = df['Cleaned Lyrics'].str.replace(r"[.,?!();:]", "")
    
    # Replace two spaces with one space
    df['Cleaned Lyrics'] = df['Cleaned Lyrics'].str.replace(r"  ", " ")
    
    # Lower-case all words
    df['Cleaned Lyrics'] = df['Cleaned Lyrics'].str.lower()
    
    # Upper-case the pronouns I -- this helps with Spacy Lemmatization
    df['Cleaned Lyrics'] = df['Cleaned Lyrics'].str.replace(r" i i ", " I I ")
    df['Cleaned Lyrics'] = df['Cleaned Lyrics'].str.replace(r" i ", " I ")
    df['Cleaned Lyrics'] = df['Cleaned Lyrics'].str.replace(r" i'", " I'")
    
    return df['Cleaned Lyrics']

```


```
def remove_song_quotes(song_row):
    """
    Removing any quotation marks around songs.
    """      
    
    # Check if the song names have quotes around them (mine did)
    try:
        song_row = re.sub(r'"', '', song_row)
    except:
        pass
    
    return song_row
```


```
def nlp_pipeline(text_document):
    """
    Takes in a string and runs it through Spacy's NLP pipeline consisting of a 
    Tokenizer, Tagger, Dependency Parser, Entity Recognizer, Text Categorizer.
    NLP features are then extracted from each Token's Spacy attributes.
    NLP features are then aggregated for the entire text_document and returned in a dictionary.
    
    Inputs:
        text_document (str): Text data. In this case it is Beatles lyrics.
    
    Output:
        dict: Aggregated NLP features.
    """
    
    # Create Spacy NLP pipeline
    nlp = en_core_web_sm.load() 
    
    # Check if text is not a string, if so return np.nan
    if type(text_document) != str:
        return np.nan
    
    # Run Spacy NLP pipeline on text_document, creates DOC object filled with tokens
    doc = nlp(text_document)

    # Tokenization
    tokens = [tok.orth_ for tok in doc]

    # Lemmatization
    lemmas = [tok.lemma_ for tok in doc]

    # Create counter dictionaries to collect counts of NLP features
    pos_counter = {}   # Coarse-grained part-of-speech
    tag_counter = {}   # Fine-grained part-of-speech.
    stop_counter = {}   # Stop word or not
    dep_counter = {}   # Syntactic dependency relation

    # Loop through each Token object contained in Doc object
    for token in doc:

        # Add token 'POS' to dictionary
        pos = "pos_" + token.pos_
        if pos in pos_counter.keys():
            pos_counter[pos] += 1
        else:
            pos_counter[pos] = 1

        # Add token 'TAG' to dictionary
        tag = "tag_" + token.tag_
        if tag in tag_counter.keys():
            tag_counter[tag] += 1
        else:
            tag_counter[tag] = 1

        # Add token 'STOP' to dictionary
        stop = "stop_" + str(token.is_stop)
        if stop in stop_counter.keys():
            stop_counter[stop] += 1
        else:
            stop_counter[stop] = 1

        # Add token 'DEP' to dictionary
        dep = "dep_" + token.dep_
        if dep in dep_counter.keys():
            dep_counter[dep] += 1
        else:
            dep_counter[dep] = 1

    # Combine NLP features into one dictionary
    nlp_dictionary = {'pos_counter' : pos_counter,
                      'tag_counter' : tag_counter,
                      'stop_counter' : stop_counter,
                      'dep_counter' : dep_counter,
                      'tokens' : tokens,
                      'lemmas' : lemmas}
    
    return nlp_dictionary
```


```
def nlp_dict_to_df(nlp_features, feature):
    """
    Take in nlp_features dictionary, outputs a pd.Series object 
    containing the NLP features.
    
    Inputs:
        nlp_features (dict): Aggregated NLP features.
    
    Outputs:
        nlp_series (pd.Series): NLP features for each song obtained using Spacy.
    """
    
    # Dep dictionary
    nlp_dict = nlp_features[feature]

    # Total number of entries in dep dictionary
    nlp_total = np.sum(list(nlp_dict.values()))

    # Calculate fraction of total for each NLP feature (Normalization)
    nlp_dict_fractions = {k: v / nlp_total for k, v in nlp_dict.items()}

    # Turn into a pandas Series to return
    nlp_series = pd.Series(nlp_dict_fractions)
    
    return nlp_series
```


```
# we have a new dataframe with cleaned lyrics
new_df = preprocessing_songs(data)
```


```
new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Lead Vocal(s)</th>
      <th>Year</th>
      <th>Lyric Line</th>
      <th>Number Lines</th>
      <th>Number Words</th>
      <th>Average Words Per Line</th>
      <th>Number Apostrophes</th>
      <th>Average Words Per Apostrophe</th>
      <th>Cleaned Lyrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Words are flowing out like endless rain into a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>words are flowing out like endless rain into a...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>They slither while they pass, they slip away a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>they slither while they pass they slip away ac...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Pools of sorrow, waves of joy are drifting thr...</td>
      <td>1</td>
      <td>12</td>
      <td>12.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>pools of sorrow waves of joy are drifting thro...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Possessing and caressing me.</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>possessing and caressing me</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Jai Guru Deva Om</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>jai guru deva om</td>
    </tr>
  </tbody>
</table>
</div>




```
# we will add all of the NLP features to the dataframe
new_df = pre_processing_songs_nlp(new_df)
```


```
spacy_df = nlp_features(new_df)
```

    100%|██████████| 5664/5664 [00:02<00:00, 2174.94it/s]
    100%|██████████| 5664/5664 [00:02<00:00, 2303.82it/s]
    100%|██████████| 5664/5664 [00:02<00:00, 2196.54it/s]
    100%|██████████| 5664/5664 [00:02<00:00, 2300.75it/s]



```
new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Lead Vocal(s)</th>
      <th>Year</th>
      <th>Lyric Line</th>
      <th>Number Lines</th>
      <th>Number Words</th>
      <th>Average Words Per Line</th>
      <th>Number Apostrophes</th>
      <th>Average Words Per Apostrophe</th>
      <th>Cleaned Lyrics</th>
      <th>NLP Features</th>
      <th>Tokens</th>
      <th>Lemmas</th>
      <th>Lemmas Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Words are flowing out like endless rain into a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>words are flowing out like endless rain into a...</td>
      <td>{'pos_counter': {'pos_NOUN': 4, 'pos_VERB': 2,...</td>
      <td>[words, are, flowing, out, like, endless, rain...</td>
      <td>[word, be, flow, out, like, endless, rain, int...</td>
      <td>word be flow out like endless rain into a pape...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>They slither while they pass, they slip away a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>they slither while they pass they slip away ac...</td>
      <td>{'pos_counter': {'pos_PRON': 3, 'pos_VERB': 3,...</td>
      <td>[they, slither, while, they, pass, they, slip,...</td>
      <td>[-PRON-, slither, while, -PRON-, pass, -PRON-,...</td>
      <td>-PRON- slither while -PRON- pass -PRON- slip a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Pools of sorrow, waves of joy are drifting thr...</td>
      <td>1</td>
      <td>12</td>
      <td>12.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>pools of sorrow waves of joy are drifting thro...</td>
      <td>{'pos_counter': {'pos_NOUN': 5, 'pos_ADP': 3, ...</td>
      <td>[pools, of, sorrow, waves, of, joy, are, drift...</td>
      <td>[pool, of, sorrow, wave, of, joy, be, drift, t...</td>
      <td>pool of sorrow wave of joy be drift through -P...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Possessing and caressing me.</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>possessing and caressing me</td>
      <td>{'pos_counter': {'pos_VERB': 2, 'pos_CCONJ': 1...</td>
      <td>[possessing, and, caressing, me]</td>
      <td>[possess, and, caress, -PRON-]</td>
      <td>possess and caress -PRON-</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Jai Guru Deva Om</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>jai guru deva om</td>
      <td>{'pos_counter': {'pos_NOUN': 4}, 'tag_counter'...</td>
      <td>[jai, guru, deva, om]</td>
      <td>[jai, guru, deva, om]</td>
      <td>jai guru deva om</td>
    </tr>
  </tbody>
</table>
</div>




```
spacy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stop_False</th>
      <th>stop_True</th>
      <th>pos_NOUN</th>
      <th>pos_VERB</th>
      <th>pos_PART</th>
      <th>pos_ADP</th>
      <th>pos_ADJ</th>
      <th>pos_DET</th>
      <th>pos_PRON</th>
      <th>pos_ADV</th>
      <th>pos_CCONJ</th>
      <th>pos_NUM</th>
      <th>pos_AUX</th>
      <th>pos_INTJ</th>
      <th>pos_X</th>
      <th>pos_PUNCT</th>
      <th>pos_PROPN</th>
      <th>pos_SYM</th>
      <th>pos_SPACE</th>
      <th>tag_NNS</th>
      <th>tag_VBP</th>
      <th>tag_VBG</th>
      <th>tag_RP</th>
      <th>tag_IN</th>
      <th>tag_JJ</th>
      <th>tag_NN</th>
      <th>tag_DT</th>
      <th>tag_PRP</th>
      <th>tag_RB</th>
      <th>tag_PRP$</th>
      <th>tag_VBN</th>
      <th>tag_CC</th>
      <th>tag_VBZ</th>
      <th>tag_TO</th>
      <th>tag_VB</th>
      <th>tag_WDT</th>
      <th>tag_CD</th>
      <th>tag_WRB</th>
      <th>tag_VBD</th>
      <th>tag_MD</th>
      <th>...</th>
      <th>dep_amod</th>
      <th>dep_pobj</th>
      <th>dep_det</th>
      <th>dep_compound</th>
      <th>dep_mark</th>
      <th>dep_advcl</th>
      <th>dep_ccomp</th>
      <th>dep_advmod</th>
      <th>dep_poss</th>
      <th>dep_cc</th>
      <th>dep_conj</th>
      <th>dep_dobj</th>
      <th>dep_nsubjpass</th>
      <th>dep_auxpass</th>
      <th>dep_xcomp</th>
      <th>dep_relcl</th>
      <th>dep_nummod</th>
      <th>dep_oprd</th>
      <th>dep_acl</th>
      <th>dep_attr</th>
      <th>dep_acomp</th>
      <th>dep_intj</th>
      <th>dep_npadvmod</th>
      <th>dep_dative</th>
      <th>dep_predet</th>
      <th>dep_neg</th>
      <th>dep_case</th>
      <th>dep_pcomp</th>
      <th>dep_quantmod</th>
      <th>dep_meta</th>
      <th>dep_subtok</th>
      <th>dep_expl</th>
      <th>dep_dep</th>
      <th>dep_punct</th>
      <th>dep_parataxis</th>
      <th>dep_nmod</th>
      <th>dep_appos</th>
      <th>dep_agent</th>
      <th>dep_csubj</th>
      <th>dep_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.636364</td>
      <td>0.363636</td>
      <td>0.363636</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.090909</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.454545</td>
      <td>0.545455</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.272727</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.583333</td>
      <td>0.416667</td>
      <td>0.416667</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.083333</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 104 columns</p>
</div>




```
new_df.to_csv('data.csv')
```


```
spacy_df.to_csv('spacy.csv')
```

# Modeling


```
data = new_df
spacy = spacy_df
```


```
# Checking for class imbalance
data['Songwriter(s)'].value_counts()
```




    Multiple Beatles    1927
    Lennon              1629
    McCartney           1488
    Harrison             612
    Starr                  8
    Name: Songwriter(s), dtype: int64




```
data[data['Songwriter(s)'] == 'Starr']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Lead Vocal(s)</th>
      <th>Year</th>
      <th>Lyric Line</th>
      <th>Number Lines</th>
      <th>Number Words</th>
      <th>Average Words Per Line</th>
      <th>Number Apostrophes</th>
      <th>Average Words Per Apostrophe</th>
      <th>Cleaned Lyrics</th>
      <th>NLP Features</th>
      <th>Tokens</th>
      <th>Lemmas</th>
      <th>Lemmas Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4495</th>
      <td>Taking a Trip to Carolina</td>
      <td>Let It Be... Naked - Fly on the Wall bonus disc</td>
      <td>Starr</td>
      <td>Starr</td>
      <td>1969</td>
      <td>Taking a trip on an ocean liner</td>
      <td>1</td>
      <td>7</td>
      <td>7.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>taking a trip on an ocean liner</td>
      <td>{'pos_counter': {'pos_VERB': 1, 'pos_DET': 2, ...</td>
      <td>[taking, a, trip, on, an, ocean, liner]</td>
      <td>[take, a, trip, on, an, ocean, liner]</td>
      <td>take a trip on an ocean liner</td>
    </tr>
    <tr>
      <th>4496</th>
      <td>Taking a Trip to Carolina</td>
      <td>Let It Be... Naked - Fly on the Wall bonus disc</td>
      <td>Starr</td>
      <td>Starr</td>
      <td>1969</td>
      <td>I'm gonna get to Carolina</td>
      <td>1</td>
      <td>5</td>
      <td>5.0</td>
      <td>1</td>
      <td>5.0</td>
      <td>i'm gonna get to carolina</td>
      <td>{'pos_counter': {'pos_PRON': 1, 'pos_VERB': 3,...</td>
      <td>[i, 'm, gon, na, get, to, carolina]</td>
      <td>[-PRON-, be, go, to, get, to, carolina]</td>
      <td>-PRON- be go to get to carolina</td>
    </tr>
    <tr>
      <th>4497</th>
      <td>Taking a Trip to Carolina</td>
      <td>Let It Be... Naked - Fly on the Wall bonus disc</td>
      <td>Starr</td>
      <td>Starr</td>
      <td>1969</td>
      <td>Taking a trip on an ocean liner</td>
      <td>1</td>
      <td>7</td>
      <td>7.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>taking a trip on an ocean liner</td>
      <td>{'pos_counter': {'pos_VERB': 1, 'pos_DET': 2, ...</td>
      <td>[taking, a, trip, on, an, ocean, liner]</td>
      <td>[take, a, trip, on, an, ocean, liner]</td>
      <td>take a trip on an ocean liner</td>
    </tr>
    <tr>
      <th>4498</th>
      <td>Taking a Trip to Carolina</td>
      <td>Let It Be... Naked - Fly on the Wall bonus disc</td>
      <td>Starr</td>
      <td>Starr</td>
      <td>1969</td>
      <td>I'm gonna get to Carolina</td>
      <td>1</td>
      <td>5</td>
      <td>5.0</td>
      <td>1</td>
      <td>5.0</td>
      <td>i'm gonna get to carolina</td>
      <td>{'pos_counter': {'pos_PRON': 1, 'pos_VERB': 3,...</td>
      <td>[i, 'm, gon, na, get, to, carolina]</td>
      <td>[-PRON-, be, go, to, get, to, carolina]</td>
      <td>-PRON- be go to get to carolina</td>
    </tr>
    <tr>
      <th>4499</th>
      <td>Taking a Trip to Carolina</td>
      <td>Let It Be... Naked - Fly on the Wall bonus disc</td>
      <td>Starr</td>
      <td>Starr</td>
      <td>1969</td>
      <td>Taking a trip on an ocean liner</td>
      <td>1</td>
      <td>7</td>
      <td>7.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>taking a trip on an ocean liner</td>
      <td>{'pos_counter': {'pos_VERB': 1, 'pos_DET': 2, ...</td>
      <td>[taking, a, trip, on, an, ocean, liner]</td>
      <td>[take, a, trip, on, an, ocean, liner]</td>
      <td>take a trip on an ocean liner</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>Taking a Trip to Carolina</td>
      <td>Let It Be... Naked - Fly on the Wall bonus disc</td>
      <td>Starr</td>
      <td>Starr</td>
      <td>1969</td>
      <td>I'm gonna get to Carolina</td>
      <td>1</td>
      <td>5</td>
      <td>5.0</td>
      <td>1</td>
      <td>5.0</td>
      <td>i'm gonna get to carolina</td>
      <td>{'pos_counter': {'pos_PRON': 1, 'pos_VERB': 3,...</td>
      <td>[i, 'm, gon, na, get, to, carolina]</td>
      <td>[-PRON-, be, go, to, get, to, carolina]</td>
      <td>-PRON- be go to get to carolina</td>
    </tr>
    <tr>
      <th>4501</th>
      <td>Taking a Trip to Carolina</td>
      <td>Let It Be... Naked - Fly on the Wall bonus disc</td>
      <td>Starr</td>
      <td>Starr</td>
      <td>1969</td>
      <td>Taking a trip on an ocean liner</td>
      <td>1</td>
      <td>7</td>
      <td>7.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>taking a trip on an ocean liner</td>
      <td>{'pos_counter': {'pos_VERB': 1, 'pos_DET': 2, ...</td>
      <td>[taking, a, trip, on, an, ocean, liner]</td>
      <td>[take, a, trip, on, an, ocean, liner]</td>
      <td>take a trip on an ocean liner</td>
    </tr>
    <tr>
      <th>4502</th>
      <td>Taking a Trip to Carolina</td>
      <td>Let It Be... Naked - Fly on the Wall bonus disc</td>
      <td>Starr</td>
      <td>Starr</td>
      <td>1969</td>
      <td>I'm gonna get to Carolina</td>
      <td>1</td>
      <td>5</td>
      <td>5.0</td>
      <td>1</td>
      <td>5.0</td>
      <td>i'm gonna get to carolina</td>
      <td>{'pos_counter': {'pos_PRON': 1, 'pos_VERB': 3,...</td>
      <td>[i, 'm, gon, na, get, to, carolina]</td>
      <td>[-PRON-, be, go, to, get, to, carolina]</td>
      <td>-PRON- be go to get to carolina</td>
    </tr>
  </tbody>
</table>
</div>



We're going to drop Ringo from the list. Sorry Ringo, poor lad.


```
data = data.drop([4495, 4496, 4497, 4498, 4499, 4500])

```


```
spacy = spacy.drop([4495, 4496, 4497, 4498, 4499, 4500])
```


```
data = data.drop([4501, 4502])
spacy = spacy.drop([4501, 4502])
```


```
data = data.reset_index(drop=True)
spacy = spacy.reset_index(drop=True)
```


```
data = data.drop("Unnamed: 0", axis=1)
```


```
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Lead Vocal(s)</th>
      <th>Year</th>
      <th>Lyric Line</th>
      <th>Number Lines</th>
      <th>Number Words</th>
      <th>Average Words Per Line</th>
      <th>Number Apostrophes</th>
      <th>Average Words Per Apostrophe</th>
      <th>Cleaned Lyrics</th>
      <th>NLP Features</th>
      <th>Tokens</th>
      <th>Lemmas</th>
      <th>Lemmas Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Words are flowing out like endless rain into a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>words are flowing out like endless rain into a...</td>
      <td>{'pos_counter': {'pos_NOUN': 4, 'pos_VERB': 2,...</td>
      <td>[words, are, flowing, out, like, endless, rain...</td>
      <td>[word, be, flow, out, like, endless, rain, int...</td>
      <td>word be flow out like endless rain into a pape...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>They slither while they pass, they slip away a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>they slither while they pass they slip away ac...</td>
      <td>{'pos_counter': {'pos_PRON': 3, 'pos_VERB': 3,...</td>
      <td>[they, slither, while, they, pass, they, slip,...</td>
      <td>[-PRON-, slither, while, -PRON-, pass, -PRON-,...</td>
      <td>-PRON- slither while -PRON- pass -PRON- slip a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Pools of sorrow, waves of joy are drifting thr...</td>
      <td>1</td>
      <td>12</td>
      <td>12.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>pools of sorrow waves of joy are drifting thro...</td>
      <td>{'pos_counter': {'pos_NOUN': 5, 'pos_ADP': 3, ...</td>
      <td>[pools, of, sorrow, waves, of, joy, are, drift...</td>
      <td>[pool, of, sorrow, wave, of, joy, be, drift, t...</td>
      <td>pool of sorrow wave of joy be drift through -P...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Possessing and caressing me.</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>possessing and caressing me</td>
      <td>{'pos_counter': {'pos_VERB': 2, 'pos_CCONJ': 1...</td>
      <td>[possessing, and, caressing, me]</td>
      <td>[possess, and, caress, -PRON-]</td>
      <td>possess and caress -PRON-</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Jai Guru Deva Om</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>jai guru deva om</td>
      <td>{'pos_counter': {'pos_NOUN': 4}, 'tag_counter'...</td>
      <td>[jai, guru, deva, om]</td>
      <td>[jai, guru, deva, om]</td>
      <td>jai guru deva om</td>
    </tr>
  </tbody>
</table>
</div>




```
spacy = spacy.drop("Unnamed: 0", axis=1)
```


```
spacy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stop_False</th>
      <th>stop_True</th>
      <th>pos_NOUN</th>
      <th>pos_VERB</th>
      <th>pos_PART</th>
      <th>pos_ADP</th>
      <th>pos_ADJ</th>
      <th>pos_DET</th>
      <th>pos_PRON</th>
      <th>pos_ADV</th>
      <th>pos_CCONJ</th>
      <th>pos_NUM</th>
      <th>pos_AUX</th>
      <th>pos_INTJ</th>
      <th>pos_X</th>
      <th>pos_PUNCT</th>
      <th>pos_PROPN</th>
      <th>pos_SYM</th>
      <th>pos_SPACE</th>
      <th>tag_NNS</th>
      <th>tag_VBP</th>
      <th>tag_VBG</th>
      <th>tag_RP</th>
      <th>tag_IN</th>
      <th>tag_JJ</th>
      <th>tag_NN</th>
      <th>tag_DT</th>
      <th>tag_PRP</th>
      <th>tag_RB</th>
      <th>tag_PRP$</th>
      <th>tag_VBN</th>
      <th>tag_CC</th>
      <th>tag_VBZ</th>
      <th>tag_TO</th>
      <th>tag_VB</th>
      <th>tag_WDT</th>
      <th>tag_CD</th>
      <th>tag_WRB</th>
      <th>tag_VBD</th>
      <th>tag_MD</th>
      <th>...</th>
      <th>dep_amod</th>
      <th>dep_pobj</th>
      <th>dep_det</th>
      <th>dep_compound</th>
      <th>dep_mark</th>
      <th>dep_advcl</th>
      <th>dep_ccomp</th>
      <th>dep_advmod</th>
      <th>dep_poss</th>
      <th>dep_cc</th>
      <th>dep_conj</th>
      <th>dep_dobj</th>
      <th>dep_nsubjpass</th>
      <th>dep_auxpass</th>
      <th>dep_xcomp</th>
      <th>dep_relcl</th>
      <th>dep_nummod</th>
      <th>dep_oprd</th>
      <th>dep_acl</th>
      <th>dep_attr</th>
      <th>dep_acomp</th>
      <th>dep_intj</th>
      <th>dep_npadvmod</th>
      <th>dep_dative</th>
      <th>dep_predet</th>
      <th>dep_neg</th>
      <th>dep_case</th>
      <th>dep_pcomp</th>
      <th>dep_quantmod</th>
      <th>dep_meta</th>
      <th>dep_subtok</th>
      <th>dep_expl</th>
      <th>dep_dep</th>
      <th>dep_punct</th>
      <th>dep_parataxis</th>
      <th>dep_nmod</th>
      <th>dep_appos</th>
      <th>dep_agent</th>
      <th>dep_csubj</th>
      <th>dep_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.636364</td>
      <td>0.363636</td>
      <td>0.363636</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.090909</td>
      <td>0.181818</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.454545</td>
      <td>0.545455</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.272727</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.272727</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.583333</td>
      <td>0.416667</td>
      <td>0.416667</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.083333</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 104 columns</p>
</div>




```
data['Songwriter(s)'].value_counts()
```




    Multiple Beatles    1927
    Lennon              1629
    McCartney           1488
    Harrison             612
    Name: Songwriter(s), dtype: int64




```
print(data.shape)
print(spacy.shape)

```

    (5656, 16)
    (5656, 104)


### Bag of Words


```
import itertools
```


```
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```


```
data['Lemmas Text']
```




    0       word be flow out like endless rain into a pape...
    1       -PRON- slither while -PRON- pass -PRON- slip a...
    2       pool of sorrow wave of joy be drift through -P...
    3                               possess and caress -PRON-
    4                                        jai guru deva om
                                  ...                        
    5651                             " love will find a way "
    5652                        gather round all -PRON- clown
    5653                           let -PRON- hear -PRON- say
    5654         hey -PRON- have get to hide -PRON- love away
    5655         hey -PRON- have get to hide -PRON- love away
    Name: Lemmas Text, Length: 5656, dtype: object




```
# Define X and y
y = data['Songwriter(s)']
X = data['Lemmas Text']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state = 444)

# Vocabulary of lemmas, use with CountVectorizer
vocabulary = set(itertools.chain.from_iterable(data['Lemmas']))

# Count vectorizer and the index-to-word mapping
count_vectorizer = CountVectorizer(vocabulary=vocabulary)

# Create bag-of-word embeddings using vectorizer
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Convert to a pandas dataframe
X_train_counts = pd.DataFrame(X_train_counts.todense())
X_test_counts = pd.DataFrame(X_test_counts.todense())

# Create a mapping from index-to-word for the bag of words
index_to_word = {v:k for k,v in count_vectorizer.vocabulary_.items()}
```


```
X_train_counts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>...</th>
      <th>1830</th>
      <th>1831</th>
      <th>1832</th>
      <th>1833</th>
      <th>1834</th>
      <th>1835</th>
      <th>1836</th>
      <th>1837</th>
      <th>1838</th>
      <th>1839</th>
      <th>1840</th>
      <th>1841</th>
      <th>1842</th>
      <th>1843</th>
      <th>1844</th>
      <th>1845</th>
      <th>1846</th>
      <th>1847</th>
      <th>1848</th>
      <th>1849</th>
      <th>1850</th>
      <th>1851</th>
      <th>1852</th>
      <th>1853</th>
      <th>1854</th>
      <th>1855</th>
      <th>1856</th>
      <th>1857</th>
      <th>1858</th>
      <th>1859</th>
      <th>1860</th>
      <th>1861</th>
      <th>1862</th>
      <th>1863</th>
      <th>1864</th>
      <th>1865</th>
      <th>1866</th>
      <th>1867</th>
      <th>1868</th>
      <th>1869</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4519</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4520</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4521</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4522</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4523</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4524 rows × 1870 columns</p>
</div>



### Latent Semantic Analysis of BoW Embedding


```
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder

```


```
le = LabelEncoder()
le.fit(y_train)
y_train_encoded = le.transform(y_train)
```


```
def plot_LSA(test_data, test_labels, plot=True):
    """
    This function first uses SK-Learn's truncated SVD (LSA) class to 
    transform the high dimensionality (number of columns) of the BoW 
    embedding down to 2 dimensions. Then the two dimensions are used
    to plot each song, colored by the song writer (class).
    
    Inputs:
        test_data (pd.DataFrame): BoW embeddings.
        test_labels (pd.Series): In this case the songwriter of each
        Beatles' song.
        plot (boolean): Whether or not to plot. Defaults to True.
    
    Outputs:
        None.
    """
    
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['blue','green','purple', 'red']
    if plot:
        plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
        blue_patch = mpatches.Patch(color='blue', label='Harrison')
        green_patch = mpatches.Patch(color='green', label='Lennon')
        purple_patch = mpatches.Patch(color='purple', label='McCartney')
        orange_patch = mpatches.Patch(color='red', label='Multiple Beatles')
        plt.legend(handles=[blue_patch, green_patch, purple_patch, orange_patch], prop={'size': 18})
        plt.xlabel('Principal Component One')
        plt.ylabel('Principal Component Two')
        plt.rcParams["xtick.labelsize"] = 20
        plt.rcParams["ytick.labelsize"] = 20



```


```
# Run LSA on BoW embeddings and plot it
fig = plt.figure(figsize=(12, 12))
plot_LSA(X_train_counts, y_train_encoded)
plt.title('LSA: BoW', fontsize = 25)
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.labelsize"] = 20
plt.xlim(-0.2, 3.2)
plt.ylim(-0.04, 0.06)
plt.show()
```


![png](images/Beatles%20NL_66_0.png)


### Latent Semantic Analysis of BoW Embedding + TFIDF


```
y = data['Songwriter(s)']
X = data['Lemmas Text']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 444)

vocabulary = set(itertools.chain.from_iterable(data['Lemmas']))

tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

X_train_tfidf = pd.DataFrame(X_train_tfidf.todense())
X_test_tfidf = pd.DataFrame(X_test_tfidf.todense())

index_to_word = {v:k for k, v in count_vectorizer.vocabulary_.items()}
```


```
fig = plt.figure(figsize=(12,12))
plot_LSA(X_train_tfidf, y_train_encoded)
plt.title('LSA: BoW + TFIDF Transformation', fontsize= 25)
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.labelsize"] = 20
plt.show()
```


![png](images/Beatles%20NL_69_0.png)


### Latent Semantic Analysis: Spacy NLP Features


```
y = data['Songwriter(s)']
X = spacy

le = LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)


```


```
fig = plt.figure(figsize=(12, 12))
plot_LSA(X, y_encoded)
plt.title('LSA: Spacy NLP Features', fontsize = 25)
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.labelsize"] = 20
plt.xlim(0.5, 1.2)
plt.show()
```


![png](images/Beatles%20NL_72_0.png)


## Modeling

We will use two machine learning algorithms to create models that will predict the songwriters of the various songs using only the lyrics as the independent variable. Then we'll use the get_most_important_features() function to pinpoint the most important words for each songwriter.


```
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
```


```
y = data['Songwriter(s)']
X = data['Cleaned Lyrics']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 444)

vocabulary = set(itertools.chain.from_iterable(data['Cleaned Lyrics']))

count_vectorizer = CountVectorizer(vocabulary=vocabulary)
```

### Logistic Regression Grid Search Pipeline


```
from sklearn.linear_model import LogisticRegression
```


```
!pip3 install pactools
```


```
from pactools.grid_search import GridSearchCVProgressBar
```


```

```

### We want to optimize to find the best estimator parameters


```


# Pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression()),])


# pipeline uses intermediate steps of CountVectorizer() and TfidfTransformer() before finally applying log regression

# This pipeline will be used as the estimator in the GridSearchCV


# Parameter Grid
param_grid = {
        'vect__max_df': (0.05, 0.075, 0.1, 0.15, 0.20, 0.25, 0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'vect__stop_words' : (None, 'english'),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


# Create the GridSearchCV object
#grid = GridSearchCV(pipeline, cv=10, n_jobs=-1, param_grid=param_grid, scoring=make_scorer(accuracy_score))
grid = GridSearchCVProgressBar(pipeline, cv=2, n_jobs=-1, param_grid=param_grid, scoring=make_scorer(accuracy_score))


# Run the grid search 
grid.fit(X_train, y_train)

# Predicts usings the best parameters of the grid search
y_pred = grid.predict(X_test)

# Accuracy Score
print('Accuracy Score:')
print(accuracy_score(y_test, y_pred))
print()

# Classification Report
print(classification_report(y_test, y_pred, digits=3))
print()

# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print()

# Best parameters from the grid search
print('Best parameters set found on development set:')
print(grid.best_params_)
print()
```


```
 def get_most_important_features(vectorizer, model, n=5):
    """
    This function is used to find the most important ngrams (words) 
    for classifying each class (songwriter).
    
    Args:
        vectorizer (sklearn.feature_extraction.text.CountVectorizer):
        The vectorizer used to create the Bag of Words space matrix.
        model (sklearn.linear_model.logistic.LogisticRegression):
        The LR model trained on the dataset.
        n (int): Number of top words to return.
    
    Returns:
        important_words (dict): Dicitionary containing the most
        critical words used to predict each songwriter class.
    """
    
    # Create a mapping dictionary from words to index
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    important_words = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        important_words[class_index] = {'tops':tops, 'bottom':bottom}
    
    return important_words



```

Running Logistic Regression and finding the most important words


```
# Vectorize
vectorizer = CountVectorizer(max_df = 0.1, ngram_range = (1, 2))
X_train_counts = vectorizer.fit_transform(X_train).todense()
X_test_counts = vectorizer.transform(X_test).todense()

# Tfidf transformation
tfidf = TfidfTransformer(norm = 'l2', use_idf = False)
X_train_counts_tfidf = tfidf.fit_transform(X_train_counts).todense()
X_test_counts_tfidf = tfidf.fit_transform(X_test_counts).todense()

clf = LogisticRegression(C = 10)
clf.fit(X_train_counts_tfidf, y_train)

# Create predictions, using trained model
y_pred = clf.predict(X_test_counts_tfidf)

# Find the most important words used for classification
importance = get_most_important_features(vectorizer, clf, 10)

# Re-name the dictionary keys
importance['Harrison'] = importance.pop(0)
importance['Lennon'] = importance.pop(1)
importance['McCartney'] = importance.pop(2)
importance['Multiple Beatles'] = importance.pop(3)

# Print the most important words
importance
```

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)





    {'Harrison': {'bottom': [(-2.045166065273923, 'cry'),
       (-2.0516036163996256, 'his'),
       (-2.074373830608004, 'down'),
       (-2.12649886774832, 'to make'),
       (-2.245646233845527, 'to be'),
       (-2.334401001365231, 'man'),
       (-2.603412111344155, 'yeah'),
       (-2.7788114268239608, 'better'),
       (-2.8376165191451923, 'but it'),
       (-3.245071662756538, 'he')],
      'tops': [(3.20668293232807, 'where you'),
       (3.2813107471829204, 'baby in'),
       (3.4526598680364975, 'it all'),
       (3.5652237899352883, 'really'),
       (3.654003225292855, 'too much'),
       (3.9902224848198466, 'got time'),
       (4.20846845653345, 'without'),
       (4.357034252936614, 'need you'),
       (4.579932179604316, 'to do'),
       (5.1964079560854515, 'me mine')]},
     'Lennon': {'bottom': [(-2.272389847767595, 'where'),
       (-2.2956971912605235, 'oh no'),
       (-2.3282633105894437, 'love you'),
       (-2.4086693661514875, 'here it'),
       (-2.4600020848978077, 'didn'),
       (-2.5905460013495665, 'know that'),
       (-2.614796732779806, 'my love'),
       (-2.838097648584486, 'without'),
       (-3.008428029862003, 'what you'),
       (-3.335549019342312, 'and she')],
      'tops': [(2.8503490406813254, 'oh now'),
       (2.9526071165781835, 'loser'),
       (3.0119376940263276, 'be alone'),
       (3.0762549486335216, 'you can'),
       (3.1254279080702903, 'nothing'),
       (3.1874394844050453, 'good night'),
       (3.205521442965275, 'well you'),
       (3.461479553947609, 'got to'),
       (3.769361331832148, 'but it'),
       (3.8967021106142603, 'get home')]},
     'McCartney': {'bottom': [(-2.1511610155645258, 'what'),
       (-2.241241980400747, 'their'),
       (-2.282678302634811, 'call'),
       (-2.4375112322430796, 'it true'),
       (-2.4586062152526638, 'me down'),
       (-2.5498405491478606, 'little girl'),
       (-2.6796571304298133, 'more'),
       (-2.7250424463530645, 'feel'),
       (-2.9858729016244396, 'think'),
       (-3.029644036108259, 'everything')],
      'tops': [(3.261969533610674, 'hello'),
       (3.359352795739651, 'want it'),
       (3.36567816522238, 'get back'),
       (3.3855530347876437, 'night before'),
       (3.4596632406431764, 'see yeah'),
       (3.5048001858749482, 'buy'),
       (3.5254689500809624, 'all my'),
       (3.6436609732156264, 'me back'),
       (3.7248591982508557, 'fast'),
       (4.011602595399383, 'and in')]},
     'Multiple Beatles': {'bottom': [(-2.380702445114838, 'can you'),
       (-2.4179787857734545, 'all you'),
       (-2.518133878680189, 'for'),
       (-2.5239313857912764, 'don'),
       (-2.662133994180265, 'me mine'),
       (-2.6709051555118153, 'around'),
       (-2.68177981063666, 'really'),
       (-2.8291554307864586, 'here'),
       (-2.892275500844382, 'still'),
       (-3.235839403441347, 'to do')],
      'tops': [(3.272339742398448, 'be your'),
       (3.3451528635041816, 'me do'),
       (3.356760928346278, 'anything'),
       (3.437673322366979, 'feeling'),
       (3.458280892290301, 'rocky'),
       (3.4692471969734586, 'know my'),
       (3.488929776316816, 'trouble'),
       (3.514775621809661, 'loves you'),
       (3.547161861254181, 'dance'),
       (3.581689596284666, 'better')]}}



### Multinomial Naive Bayes Grid Search Pipeline


```
from sklearn.naive_bayes import MultinomialNB
```


```


# Pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())])

# Parameter Grid
param_grid = {
        'vect__max_df': (0.05, 0.075, 0.1, 0.15, 0.20, 0.25, 0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'vect__stop_words' : (None, 'english'),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0, 0.001, 0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0)}

# Create the GridSearchCV object
grid = GridSearchCV(pipeline, cv=10, n_jobs=-1, param_grid=param_grid, scoring=make_scorer(accuracy_score))

# Run the grid search 
grid.fit(X_train, y_train)

# Predict usings the best parameters of the grid search
y_pred = grid.predict(X_test)

# Accuracy Score
print('Accuracy Score:')
print(accuracy_score(y_test, y_pred))
print()

# Classification Report
print(classification_report(y_test, y_pred, digits=3))
print()

# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print()

# Best parameters from the grid search
print('Best parameters set found on development set:')
print(grid.best_params_)
print()
```


```
# Vectorize
vectorizer = CountVectorizer(max_df = 0.1, ngram_range = (1, 2))
X_train_counts = vectorizer.fit_transform(X_train).todense()
X_test_counts = vectorizer.transform(X_test).todense()

# Tfidf transformation
tfidf = TfidfTransformer(norm = 'l2', use_idf = False)
X_train_counts_tfidf = tfidf.fit_transform(X_train_counts).todense()
X_test_counts_tfidf = tfidf.fit_transform(X_test_counts).todense()

nb = MultinomialNB()
nb.fit(X_train_counts_tfidf, y_train)

# Create predictions, using trained model
y_pred = nb.predict(X_test_counts_tfidf)

# Find the most important words used for classification
importance = get_most_important_features(vectorizer, nb, 10)

# Re-name the dictionary keys
importance['Harrison'] = importance.pop(0)
importance['Lennon'] = importance.pop(1)
importance['McCartney'] = importance.pop(2)
importance['Multiple Beatles'] = importance.pop(3)

# Print the most important words
importance
```




    {'Harrison': {'bottom': [(-9.377282492934983, 'your trust'),
       (-9.377282492934983, 'your voice'),
       (-9.377282492934983, 'yours sincerely'),
       (-9.377282492934983, 'yours yet'),
       (-9.377282492934983, 'yourself home'),
       (-9.377282492934983, 'yourself in'),
       (-9.377282492934983, 'yourself to'),
       (-9.377282492934983, 'zapped'),
       (-9.377282492934983, 'zapped in'),
       (-9.377282492934983, 'zoo what')],
      'tops': [(-7.106033702676536, 'not'),
       (-7.099118495814187, 'll'),
       (-7.0610869856031915, 'that'),
       (-6.910337502354965, 'if'),
       (-6.870088757600826, 'be'),
       (-6.857382826987659, 'know'),
       (-6.831224065606969, 'love'),
       (-6.545435529591376, 'don'),
       (-6.285568876248815, 'all'),
       (-6.179713804302713, 'it')]},
     'Lennon': {'bottom': [(-9.557661329838217, 'your voice'),
       (-9.557661329838217, 'your window'),
       (-9.557661329838217, 'yours sincerely'),
       (-9.557661329838217, 'yourself'),
       (-9.557661329838217, 'yourself home'),
       (-9.557661329838217, 'yourself in'),
       (-9.557661329838217, 'yourself no'),
       (-9.557661329838217, 'yourself to'),
       (-9.557661329838217, 'zoo'),
       (-9.557661329838217, 'zoo what')],
      'tops': [(-6.234386539162086, 'in'),
       (-6.226543065041415, 'all'),
       (-6.226129185516754, 'that'),
       (-6.2054850694945145, 'know'),
       (-6.19591051876331, 'my'),
       (-6.188792385169995, 'can'),
       (-6.122542730040965, 'is'),
       (-6.01312671508577, 'love'),
       (-5.973525540057158, 'she'),
       (-5.684910114649124, 'it')]},
     'McCartney': {'bottom': [(-9.53302596278112, 'your vest'),
       (-9.53302596278112, 'your way'),
       (-9.53302596278112, 'your window'),
       (-9.53302596278112, 'yours yet'),
       (-9.53302596278112, 'yourself home'),
       (-9.53302596278112, 'yourself no'),
       (-9.53302596278112, 'zapped'),
       (-9.53302596278112, 'zapped in'),
       (-9.53302596278112, 'zoo'),
       (-9.53302596278112, 'zoo what')],
      'tops': [(-6.488101822072984, 'your'),
       (-6.48290034705385, 'down'),
       (-6.471984767206914, 'don'),
       (-6.446834475852382, 'my'),
       (-6.444107708474709, 'love'),
       (-6.4276420486337775, 'of'),
       (-6.372149255906219, 'she'),
       (-6.340993470408326, 'be'),
       (-6.0683605112704395, 'in'),
       (-5.89485844637437, 'it')]},
     'Multiple Beatles': {'bottom': [(-9.598712792303138, 'your taste'),
       (-9.598712792303138, 'your teddy'),
       (-9.598712792303138, 'your vest'),
       (-9.598712792303138, 'your voice'),
       (-9.598712792303138, 'yours'),
       (-9.598712792303138, 'yours sincerely'),
       (-9.598712792303138, 'yours yet'),
       (-9.598712792303138, 'yourself no'),
       (-9.598712792303138, 'zapped'),
       (-9.598712792303138, 'zapped in')],
      'tops': [(-6.170681412914195, 'yeah'),
       (-6.118398153346851, 'your'),
       (-6.016575553239925, 'all'),
       (-6.012901653314865, 'that'),
       (-5.998315932927394, 'know'),
       (-5.9692317567912685, 'be'),
       (-5.964634958768828, 'in'),
       (-5.9403216686676235, 'my'),
       (-5.881664089813732, 'it'),
       (-5.6590783376237415, 'love')]}}



# Lyric Generation


```
from google.colab import files
```


```
import os, pandas as pd
```


```
os.chdir('./gdrive/My Drive/Google Colaboratory/Colab Notebooks/Data Science   Machine Learning/Beatles NLP/Data')
```


```
data = pd.read_csv('data.csv')
```


```
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Lead Vocal(s)</th>
      <th>Year</th>
      <th>Lyric Line</th>
      <th>Number Lines</th>
      <th>Number Words</th>
      <th>Average Words Per Line</th>
      <th>Number Apostrophes</th>
      <th>Average Words Per Apostrophe</th>
      <th>Cleaned Lyrics</th>
      <th>NLP Features</th>
      <th>Tokens</th>
      <th>Lemmas</th>
      <th>Lemmas Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Words are flowing out like endless rain into a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>words are flowing out like endless rain into a...</td>
      <td>{'pos_counter': {'pos_NOUN': 4, 'pos_VERB': 2,...</td>
      <td>['words', 'are', 'flowing', 'out', 'like', 'en...</td>
      <td>['word', 'be', 'flow', 'out', 'like', 'endless...</td>
      <td>word be flow out like endless rain into a pape...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>They slither while they pass, they slip away a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>they slither while they pass they slip away ac...</td>
      <td>{'pos_counter': {'pos_PRON': 3, 'pos_VERB': 3,...</td>
      <td>['they', 'slither', 'while', 'they', 'pass', '...</td>
      <td>['-PRON-', 'slither', 'while', '-PRON-', 'pass...</td>
      <td>-PRON- slither while -PRON- pass -PRON- slip a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Pools of sorrow, waves of joy are drifting thr...</td>
      <td>1</td>
      <td>12</td>
      <td>12.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>pools of sorrow waves of joy are drifting thro...</td>
      <td>{'pos_counter': {'pos_NOUN': 5, 'pos_ADP': 3, ...</td>
      <td>['pools', 'of', 'sorrow', 'waves', 'of', 'joy'...</td>
      <td>['pool', 'of', 'sorrow', 'wave', 'of', 'joy', ...</td>
      <td>pool of sorrow wave of joy be drift through -P...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Possessing and caressing me.</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>possessing and caressing me</td>
      <td>{'pos_counter': {'pos_VERB': 2, 'pos_CCONJ': 1...</td>
      <td>['possessing', 'and', 'caressing', 'me']</td>
      <td>['possess', 'and', 'caress', '-PRON-']</td>
      <td>possess and caress -PRON-</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Jai Guru Deva Om</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>jai guru deva om</td>
      <td>{'pos_counter': {'pos_NOUN': 4}, 'tag_counter'...</td>
      <td>['jai', 'guru', 'deva', 'om']</td>
      <td>['jai', 'guru', 'deva', 'om']</td>
      <td>jai guru deva om</td>
    </tr>
  </tbody>
</table>
</div>




```
data.rename(columns={'Unnamed: 0':'Line ID'}, inplace=True)
```


```
list(data)
```




    ['Line ID',
     'Song',
     'Album Debut',
     'Songwriter(s)',
     'Lead Vocal(s)',
     'Year',
     'Lyric Line',
     'Number Lines',
     'Number Words',
     'Average Words Per Line',
     'Number Apostrophes',
     'Average Words Per Apostrophe',
     'Cleaned Lyrics',
     'NLP Features',
     'Tokens',
     'Lemmas',
     'Lemmas Text']




```
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Line ID</th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Lead Vocal(s)</th>
      <th>Year</th>
      <th>Lyric Line</th>
      <th>Number Lines</th>
      <th>Number Words</th>
      <th>Average Words Per Line</th>
      <th>Number Apostrophes</th>
      <th>Average Words Per Apostrophe</th>
      <th>Cleaned Lyrics</th>
      <th>NLP Features</th>
      <th>Tokens</th>
      <th>Lemmas</th>
      <th>Lemmas Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Words are flowing out like endless rain into a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>words are flowing out like endless rain into a...</td>
      <td>{'pos_counter': {'pos_NOUN': 4, 'pos_VERB': 2,...</td>
      <td>['words', 'are', 'flowing', 'out', 'like', 'en...</td>
      <td>['word', 'be', 'flow', 'out', 'like', 'endless...</td>
      <td>word be flow out like endless rain into a pape...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>They slither while they pass, they slip away a...</td>
      <td>1</td>
      <td>11</td>
      <td>11.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>they slither while they pass they slip away ac...</td>
      <td>{'pos_counter': {'pos_PRON': 3, 'pos_VERB': 3,...</td>
      <td>['they', 'slither', 'while', 'they', 'pass', '...</td>
      <td>['-PRON-', 'slither', 'while', '-PRON-', 'pass...</td>
      <td>-PRON- slither while -PRON- pass -PRON- slip a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Pools of sorrow, waves of joy are drifting thr...</td>
      <td>1</td>
      <td>12</td>
      <td>12.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>pools of sorrow waves of joy are drifting thro...</td>
      <td>{'pos_counter': {'pos_NOUN': 5, 'pos_ADP': 3, ...</td>
      <td>['pools', 'of', 'sorrow', 'waves', 'of', 'joy'...</td>
      <td>['pool', 'of', 'sorrow', 'wave', 'of', 'joy', ...</td>
      <td>pool of sorrow wave of joy be drift through -P...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Possessing and caressing me.</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>possessing and caressing me</td>
      <td>{'pos_counter': {'pos_VERB': 2, 'pos_CCONJ': 1...</td>
      <td>['possessing', 'and', 'caressing', 'me']</td>
      <td>['possess', 'and', 'caress', '-PRON-']</td>
      <td>possess and caress -PRON-</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>Jai Guru Deva Om</td>
      <td>1</td>
      <td>4</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>jai guru deva om</td>
      <td>{'pos_counter': {'pos_NOUN': 4}, 'tag_counter'...</td>
      <td>['jai', 'guru', 'deva', 'om']</td>
      <td>['jai', 'guru', 'deva', 'om']</td>
      <td>jai guru deva om</td>
    </tr>
  </tbody>
</table>
</div>




```
lyrics_data = data[['Line ID', 'Song', 'Album Debut', 'Songwriter(s)', 'Year', 'Cleaned Lyrics']]
```


```
lyrics_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Line ID</th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Year</th>
      <th>Cleaned Lyrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>words are flowing out like endless rain into a...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>they slither while they pass they slip away ac...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>pools of sorrow waves of joy are drifting thro...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>possessing and caressing me</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>jai guru deva om</td>
    </tr>
  </tbody>
</table>
</div>




```
lyrics_data = lyrics_data.set_index('Line ID')
```


```
lyrics_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Song</th>
      <th>Album Debut</th>
      <th>Songwriter(s)</th>
      <th>Year</th>
      <th>Cleaned Lyrics</th>
    </tr>
    <tr>
      <th>Line ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>words are flowing out like endless rain into a...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>they slither while they pass they slip away ac...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>pools of sorrow waves of joy are drifting thro...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>possessing and caressing me</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Across the Universe</td>
      <td>Let It Be</td>
      <td>Lennon</td>
      <td>1968</td>
      <td>jai guru deva om</td>
    </tr>
  </tbody>
</table>
</div>




```
# We will read in all of the cleaned lyrics into a single text file for training
with open('lyricsText.txt', 'w', encoding='utf-8') as filehandle:
  for listitem in lyrics_data['Cleaned Lyrics']:
    filehandle.write('%s\n' % listitem)
```

## Using textgenrnn to generate lyrics


```
!pip install -q textgenrnn
```


```
from textgenrnn import textgenrnn
```


```
model_cfg = {
    'rnn_size': 128,
    'rnn_layers': 4,
    'rnn_bidirectional': True,
    'max_length': 40,
    'max_words': 10000,
    'dim_embeddings': 100,
    'word_level': False,
}

train_cfg = {
    'line_delimited': True,
    'num_epochs': 10, 
    'gen_epochs': 2,
    'batch_size': 1024,
    'train_size': 0.8,
    'dropout': 0.0,
    'max_gen_length': 300,
    'validation': True,
    'is_csv': False
}
```


```
text_file = 'lyricsText.txt'
```


```
model_name = '500nds_12Lrs_100epchs_Model'
```


```
model_name = '500nds_12Lrs_100epchs_Model'
textgen = textgenrnn(name=model_name)

train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file

train_function(
    file_path=text_file,
    new_model=True,
    num_epochs=train_cfg['num_epochs'],
    gen_epochs=train_cfg['gen_epochs'],
    batch_size=train_cfg['batch_size'],
    train_size=train_cfg['train_size'],
    dropout=train_cfg['dropout'],
    max_gen_length=train_cfg['max_gen_length'],
    validation=train_cfg['validation'],
    is_csv=train_cfg['is_csv'],
    rnn_layers=model_cfg['rnn_layers'],
    rnn_size=model_cfg['rnn_size'],
    rnn_bidirectional=model_cfg['rnn_bidirectional'],
    max_length=model_cfg['max_length'],
    dim_embeddings=model_cfg['dim_embeddings'],
    word_level=model_cfg['word_level'])
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:541: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4432: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:197: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:203: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:66: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/optimizers.py:793: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3576: The name tf.log is deprecated. Please use tf.math.log instead.
    
    5,663 texts collected.
    Training new model w/ 4-layer, 128-cell Bidirectional LSTMs
    Training on 134,882 character sequences.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    
    Epoch 1/10
    131/131 [==============================] - 1124s 9s/step - loss: 2.7811 - val_loss: 2.1586
    Epoch 2/10
    131/131 [==============================] - 1108s 8s/step - loss: 2.0265 - val_loss: 1.9074
    ####################
    Temperature: 0.2
    ####################
    i could the coud you cand the goon
    
    i love the with the love you cand the could to cour
    
    i whe whe with a could the she with tour the soud I cone the love and the love the soud
    
    ####################
    Temperature: 0.5
    ####################
    i coud you me with a loood the cous I will the the the shen you coul to tour
    
    wan I cand to me love the lowe I'll the criee
    
    when don't the with you cone
    
    ####################
    Temperature: 1.0
    ####################
    dacl I'ly homendo gove thete aneaybine can by choum it yis coud
    
    oh fouy coont trus
    
    she your fell the love githor eplirine
    
    Epoch 3/10
    131/131 [==============================] - 1105s 8s/step - loss: 1.7636 - val_loss: 1.6658
    Epoch 4/10
    131/131 [==============================] - 1104s 8s/step - loss: 1.5593 - val_loss: 1.5089
    ####################
    Temperature: 0.2
    ####################
    i don't go me me
    
    i'm gonna know you know you know
    
    i know you've got to be make
    
    ####################
    Temperature: 0.5
    ####################
    baby nah nah nah nah nah nah
    
    i'm grad me
    
    you're goet to show you know
    
    ####################
    Temperature: 1.0
    ####################
    and need guiltad arone
    
    she should dane
    
    hele sleep it's liel me
    
    Epoch 5/10
    131/131 [==============================] - 1103s 8s/step - loss: 1.4092 - val_loss: 1.3872
    Epoch 6/10
    131/131 [==============================] - 1103s 8s/step - loss: 1.2847 - val_loss: 1.3122
    ####################
    Temperature: 0.2
    ####################
    i want to be the life on the man
    
    i want to be the monely want
    
    i want to be the money the way
    
    ####################
    Temperature: 0.5
    ####################
    and I'll all together now
    
    i know you'll be like me a will you
    
    when I'm going to me
    
    ####################
    Temperature: 1.0
    ####################
    shece I go a carem on
    
    so child ways don't apparty can so
    
    mo the pain to but me nobody
    
    Epoch 7/10
    131/131 [==============================] - 1099s 8s/step - loss: 1.1877 - val_loss: 1.2614
    Epoch 8/10
    131/131 [==============================] - 1099s 8s/step - loss: 1.1100 - val_loss: 1.2192
    ####################
    Temperature: 0.2
    ####################
    i saw a shame what a shame mary jane baby jud
    
    i saw a show and she would say so should
    
    i want to be your man and she said
    
    ####################
    Temperature: 0.5
    ####################
    the calling and name and the party
    
    i saw a walk and I sad a show
    
    i need to see you and me mine
    
    ####################
    Temperature: 1.0
    ####################
    is well you see you
    
    there will getting better is understand
    
    but as belight up an mind after my my hand
    
    Epoch 9/10
    131/131 [==============================] - 1102s 8s/step - loss: 1.0410 - val_loss: 1.1814
    Epoch 10/10
    131/131 [==============================] - 1105s 8s/step - loss: 0.9878 - val_loss: 1.1679
    ####################
    Temperature: 0.2
    ####################
    it's going to lose that you see
    
    and I will sing the word of the same
    
    and I will send you and I'll be and I'll be there
    
    ####################
    Temperature: 0.5
    ####################
    can't you see you thangering love boy
    
    with in the evening
    
    i thought a should can't be long
    
    ####################
    Temperature: 1.0
    ####################
    oh oh ah ah oh
    
    don't let me didn't let me down
    
    look she's meeting home
    



```
print(textgen.model.summary())
```

    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input (InputLayer)              (None, 40)           0                                            
    __________________________________________________________________________________________________
    embedding (Embedding)           (None, 40, 100)      4100        input[0][0]                      
    __________________________________________________________________________________________________
    rnn_1 (Bidirectional)           (None, 40, 256)      234496      embedding[0][0]                  
    __________________________________________________________________________________________________
    rnn_2 (Bidirectional)           (None, 40, 256)      394240      rnn_1[0][0]                      
    __________________________________________________________________________________________________
    rnn_3 (Bidirectional)           (None, 40, 256)      394240      rnn_2[0][0]                      
    __________________________________________________________________________________________________
    rnn_4 (Bidirectional)           (None, 40, 256)      394240      rnn_3[0][0]                      
    __________________________________________________________________________________________________
    rnn_concat (Concatenate)        (None, 40, 1124)     0           embedding[0][0]                  
                                                                     rnn_1[0][0]                      
                                                                     rnn_2[0][0]                      
                                                                     rnn_3[0][0]                      
                                                                     rnn_4[0][0]                      
    __________________________________________________________________________________________________
    attention (AttentionWeightedAve (None, 1124)         1124        rnn_concat[0][0]                 
    __________________________________________________________________________________________________
    output (Dense)                  (None, 41)           46125       attention[0][0]                  
    ==================================================================================================
    Total params: 1,468,565
    Trainable params: 1,468,565
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None



```
weights_path = '{}_weights.hdf5'.format(model_name)

```


```
vocab_path = '500nds_12Lrs_100epchs_Model_vocab (1).json'
```


```
config_path = '500nds_12Lrs_100epchs_Model_config.json'
```


```
textgen = textgenrnn(weights_path=weights_path,
                       vocab_path=vocab_path,
                       config_path=config_path)

generated_characters = 300

text = textgen.generate_samples(300)

```

    ####################
    Temperature: 0.2
    ####################
    i want you
    
    i want to be your man I want to be your man
    
    i want you so bad
    
    you know you know my name
    
    i want you
    
    i want to be your man I me mine
    
    i want to be your man and kissing the same of the bight the sun
    
    i can't be and me mine
    
    i want you so bad
    
    you know I nearly had a pain at the party
    
    i want you so bad
    
    and I will sing the time of the same
    
    i want you so hold you
    
    i said someone who's got to be your man
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I'm true
    
    i want to be your man I'm lonely
    
    i want you so hold me too
    
    when I saw you should see you make me the mind
    
    i want you so bad
    
    i want you
    
    it's all too much the number
    
    i want you
    
    i want you so hold you man
    
    when I was you see you and me
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I know what you know my name
    
    i want you
    
    i can't be and me alone
    
    it's true another man
    
    i want you so bad
    
    i want to be your man I don't know that you see
    
    i want to be your man I me mine
    
    i want you
    
    the word the word there's no need
    
    i want you so hold you head you can do
    
    i want to be your man I can't see
    
    i want to be your man I me mine
    
    i want you so hold you
    
    i want to be your man I me mine
    
    i want you so bad
    
    i want you so bad
    
    i want you so hold me too much oh
    
    i want to be your man I me mine
    
    i want you
    
    i want you
    
    i want you so bad
    
    the word the word of the start
    
    i want you so bad
    
    i want you so hard that you see
    
    i want you so bad
    
    i want you so bad
    
    i can't believe you
    
    i want you so hold you man
    
    i want you so hold me too
    
    i want you so bad
    
    i want to be your man I me mine
    
    i want you so bad
    
    i want you so many many jane had a pain at the party
    
    i want you so bad
    
    i want you so bad
    
    when I saw you should see the end
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I me mine
    
    i want you so bad
    
    i'm gonna cread you there met me so mind
    
    i'm tried to make you mean to me
    
    i think you with another day
    
    when I was a shame what a shame mary jane had a pain at the party
    
    you know I nearly had a pain at the party
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I me mine
    
    i want you so hard the word there with a lot of the band
    
    i can't be long to be long listen play
    
    when I was you see you see you mean to me
    
    i want you so bad
    
    i want you so hold me too
    
    i want you so bad
    
    and I want to be your man 
    
    i want you so bad
    
    i want you you should see you know
    
    it's all too much you yeah
    
    i want you so bad
    
    i want you so bad
    
    i want you
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I me mine
    
    i want you so bad
    
    i want you so bad
    
    i want you
    
    i want you so hold me too
    
    i want you so hard the changes the same
    
    i want you so hold me too
    
    i want you so hard the way be baby cry
    
    i want you
    
    i want you so hold me there
    
    i think you should see you see
    
    i want to be your man I me mine
    
    i want you so hold you
    
    i want you so hard that I was a show
    
    i want you so bad
    
    when I saw you should see you want to be your man
    
    i want to be your man I me mine
    
    i want you so bad
    
    and the word the sun she was a shame
    
    i want you so bad
    
    i want you so bad
    
    all the world there's no need you
    
    i want you
    
    i want you so bad
    
    it's all too much oh yeah
    
    the world that she said so
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    i want you you see you so make me
    
    i want to be your man I me mine
    
    i want you so bad
    
    i want you so bad
    
    i'm gonna get to be there with you
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I me mine
    
    i want you
    
    i want you so bad
    
    i can't be you now the number
    
    i want you so bad
    
    i can't be and I feel and I'll love you
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I'm really down
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I'm really down
    
    i want you so bad
    
    i want you so hold you oh
    
    i want you so bad
    
    i want you so hold your mind
    
    i want you so bad
    
    and I will send the time at all the time
    
    i want to be your man why
    
    i want you so hold you
    
    the word the word there's no need
    
    i want to be your man I me mine
    
    i want you so bad
    
    i want you so hold me too much to me
    
    i want you so hold me that I'm so selly down
    
    i want you so bad
    
    i want you so bad
    
    the word the word she said a love me when you believe been to your man
    
    i want you so bad
    
    i want to be your man I can do
    
    i want you so bad
    
    i want to be your man I me mine
    
    and I want to be your man be man
    
    i want you so bad
    
    i want you so hold me too
    
    when I was you see you there metter sing
    
    i want you so bad
    
    i want to be your man I me mine
    
    i want you so bad
    
    i want you so see you to me
    
    i want you you should see you see
    
    i want you so hold you
    
    i want you
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    the world the word the sun
    
    i want you so bad
    
    i want you so bad
    
    when I was you see you believe me be long
    
    i can't be baby beep beep me the wall
    
    i want you so bad
    
    i want you so hold you
    
    i'm down I'm really down
    
    i want to be your man I me mine
    
    i want you so hold me too
    
    i want you so hold you should
    
    and I want to be your man I'm really down
    
    i want you so bad
    
    i want you so bad
    
    i want you so hold you
    
    the word the word there's no need
    
    when I say he hard and feeling here
    
    i can't be no no no no no no no
    
    i want you so bad
    
    i can't be there with a was a rich oh
    
    i want you so bad
    
    come on the beginning armary there
    
    i want you so hard that you see
    
    the world will spread the sun
    
    i want to be your man 
    
    i want you so bad
    
    i want you so bad
    
    it's all too much you
    
    i want to be your mother should
    
    i want you so hard that you know
    
    i want you
    
    i want you so bad
    
    i can't be and mine
    
    i want you so bad
    
    i want you so hold you
    
    i want you so many many many jane had a pain at the party
    
    i want you so hard me
    
    i want you so bad
    
    i want you so bad
    
    and I will see the word with a lot
    
    i want you so hard that I was a shame mary jane what a shame mary jane had a pain at the party
    
    i want you so bad
    
    i want you so cry
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I me mine
    
    and I will see the word there's no need
    
    i want to be your man I'm true
    
    i want you so bad
    
    i want to be your man I me mine
    
    i can't be long long long to be bad you belong
    
    i want you so bad
    
    and I will send the world that you see it a pain of mind
    
    i want you so hold your mind
    
    i want you you should mean to me
    
    i want to be your man I me mine
    
    i want you so bad
    
    i want you so bad
    
    i want you
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    you know I nearly hand you have and never been a long that line
    
    i want you so bad
    
    i want you you should see you mean to me
    
    it's all too much yeah
    
    she said so how me and I'll go
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    i want to be your man I can't stay
    
    and I will send the world I'm the world
    
    when I was you see you love me down
    
    and I want to be your man I'm down
    
    and I would see you to make you
    
    and I can see your money more
    
    i want you
    
    i want you so hold your mind
    
    i want you so bad
    
    and I will say the word there's no need
    
    i want you
    
    she was a shame what a shame mary jane had a pain at the party
    
    when I saw you see you know my name
    
    i want you so bad
    
    i want to be your man I want to be your man
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    i can't go that you once belonged
    
    she was a way the word of the band
    
    i want you so bad
    
    i want you so hold you
    
    i want you so bad
    
    i want you so bad
    
    the love that's all the time to say you see
    
    i want you so bad
    
    i want you so hold me too
    
    i want you so bad
    
    and I will send you to me
    
    i want to be your man I can't do that
    
    i want you so bad
    
    she said so should can be me
    
    i think you should see you see
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    i want you so bad
    
    i want you
    
    i want you so bad
    
    i want you
    
    i want to be your man I me mine
    
    i want you so hard that I was a pain to me
    
    i want you so hold you
    
    i want you so bad
    
    and I will send the time at all the time
    
    i don't know what you know my name
    
    i want you so hold you
    
    i want you so hard that head
    
    i want you so bad
    
    i want you so hard the word the same on the party
    
    the word the sun she was be baby just can't be mine
    
    i want you so bad
    
    i want to be your man I me long to you
    
    ####################
    Temperature: 0.5
    ####################
    i want you
    
    you can do that I saw you and that she said
    
    you know you know my name
    
    the bagin love will you go
    
    i want to be your studant so much in the night
    
    you can talk to much you
    
    come on when I have comes and the children
    
    and I need the end
    
    i know for the night before
    
    but I'll go the end
    
    he no one little girl
    
    we didn't need me the night before
    
    when you go arout the only window you had to go
    
    i want you you should see you be bad
    
    it could see you know I nearly buy
    
    you know what you see it a partaction
    
    i'm gonna get that you can be me
    
    i want to be your man to make it a pain at a song on the dark
    
    the word I want to be your man
    
    i saw you should see me the one to me
    
    i never been a long to be there
    
    i can't talk you away yeah yeah yeah
    
    i said and now I need you
    
    so cry many man she had had easy
    
    when I do you can see that you see
    
    and then I'm all you see
    
    tell me on love loved to be your life
    
    i said
    
    you know I'm down for your mind
    
    can't you hold me and I have's gonna be a party what you see her too
    
    love her
    
    i said she said sunshine
    
    i'm be my long to be long listen better
    
    beep me I'm down the said
    
    i would do that you've been me
    
    i could never got to see you with yeah
    
    and I'll be your man 
    
    we're gonna stays where do a way be wrong
    
    mother sump and when you know
    
    see a book and I'll love you
    
    when I get to be your man I me mine
    
    i want you
    
    all the wind I want the road
    
    i said some how what it the same
    
    it's all too hold me and I'm helalize
    
    i'm gonna do you so much yeah
    
    i'm be long you no but you'll be mine
    
    i'm right all you near you soul
    
    hello hello
    
    i want to be your man 
    
    i want you so bad
    
    i can't be good on my love that's belonged
    
    i'll get you near me the same with a was a star
    
    don't worry
    
    when I saw the man oh the same
    
    and I will want to be no oh
    
    i want to be your man come
    
    want to be where you see you need
    
    don't say here me so sad her belong
    
    she said so hold me there before to me
    
    i want you to day
    
    i want you so hard she's gonna be too much 
    
    i don't know what you see me me
    
    will you in the one on the one and in the sun
    
    i want to be your man I got to be your man
    
    i want to be your man I me mine
    
    and you know you know I think you that you know
    
    now the sun sun now there is way
    
    tell me when I am the way of mind
    
    but I'm gonna be sad you see
    
    it's all too much that like of yeah
    
    she takes a father should want
    
    you know I'm not there words of you
    
    make a way be wants to cold you
    
    baby you're so making and still the way
    
    and she world the sun I feel again
    
    there are no one love is all you need
    
    i'm gonna get that I say going to me
    
    i want to be the polled you will right
    
    i want you
    
    there's no need me to the long
    
    so on I saw was to see you can't be mine
    
    if I want you so hard to me
    
    it's all the lonely penet to me
    
    look at the old with you
    
    i want you
    
    you know that I want to be your man
    
    i said a shord what I know how may
    
    i'm looking through you where you coust to see the love is a place
    
    and I will love you you can talk to me
    
    treat me so long that have head
    
    come on the bottom love girl window
    
    when I tell you you near you don't know my nind hey by should
    
    i can't be there one on the place in the party
    
    when I wake there's on his nowhere
    
    come on the bottom things are show
    
    and the play back words of things is the sky with and everything I will do you old you can send that something I say be with a lot
    
    i want to hold your hand
    
    we'll never got a ticket to be a paperback writer
    
    i want you
    
    she was a little girl
    
    i think you what you know my name
    
    love you near me the man and there's and I feel the world
    
    the world of the sun
    
    i want you you so many touch
    
    i'm coming oo so hold me there
    
    i'm gonna be an our out her them
    
    i want you you
    
    and I will see you you can't be bad you
    
    she keeps something away
    
    let it be let it be bad out to be your mone
    
    i want you
    
    she said where you should can sleep tonight
    
    i'm down I'm so so
    
    i'm gonna get to see you more
    
    i want you so comport me so man
    
    there broken't let me down
    
    come on let's nice him time
    
    and the weater has back the start
    
    i said there's no one me what a plann
    
    i'm gonna carry that you know
    
    i can't go out you and me and I want to be your man 
    
    to carry to show you have to can all me
    
    all the lonely better girl
    
    and nothing to see you party change at his pulias
    
    i want to be your man to be head
    
    i can't believe you
    
    just come on baby true
    
    cause I hey have gone
    
    help me was it is a with you
    
    the way where are of the billed in the same
    
    she world say she's a lot
    
    sexy not a little girl from my friends
    
    and I sing a little girl
    
    say the love is here the begins
    
    me the man what a shame mary jane wanded there with you
    
    i want you
    
    the word the way will seem me
    
    yeah yeah yeah yeah yeah yeah
    
    here such a love that show what I near yeah
    
    i want you worry
    
    i want to be your man to show
    
    i don't know why hey love me shere it cry
    
    the way thing will go
    
    it's only only skenter wearing with a long to be me
    
    you know I want you
    
    all my number there's have a good from the long out of the end
    
    and I want to be the one what you lied you
    
    and I will I can see you that you know you know my milly feeling
    
    there's a got a fool on the party
    
    the world is the sky weres and so his nowhere
    
    when I can't hide nowhere and post to the way
    
    we do it is it's real
    
    i want to hold your mind
    
    you know you know my name
    
    it's so a love with her it's sleeping
    
    and I lost you you know you want me
    
    i want you
    
    when I was it the sone in the same
    
    she's the tearing all the world there are going to see
    
    i think you hear such oh
    
    i can't a chair that have across of mind
    
    they way be won't be sad feeling
    
    i want you so bad
    
    i can't be and mine
    
    all the word there's a lonely
    
    now I need to me
    
    i want you so bad
    
    you know what you say see
    
    i want you you can be your mother should
    
    yeah yeah yeah yeah yeah
    
    and I will send the only time that have can comping out the band
    
    well you can be and me my loving of when I say goodbye oh oh oh oh
    
    and she said wead too
    
    when I was you start the one
    
    i want to be your eyes can't you door
    
    she outs and heart a really one that with you
    
    it's been a long to carolle
    
    i got to be the sman I'm mad me ware
    
    i can't get it be time to me
    
    that is way there's no need
    
    and I love you
    
    good day so hold me in the fool
    
    and I get the same and still wings that you do
    
    because I'm the old you have got to have me a show
    
    the tange at all the world that you do
    
    i want you to stay that you see
    
    the calles you say that mean a just before that
    
    i want you so see her too
    
    now I'm so the door of things of mind
    
    the have what do disamet
    
    so man I need no no oh head on the end
    
    but broken the sun and I think you
    
    all together now
    
    you can't be the feet and long to you
    
    did I take you and the way and I'm gonna be long or gain on
    
    i want you so once paperback
    
    the world that I still you
    
    she said hello changile girl like a parter
    
    so the sun going for your door
    
    i can't believe me the same long to be make her me
    
    yes I'm so leave it all too much me
    
    the sun people better here
    
    now I need to see the only girl to me
    
    and I loved you
    
    that you see it and I'll try too
    
    hey gonna be a word of long
    
    well you near me the word is that there
    
    everybody can say him tance and I don't know
    
    i want you
    
    i know what I just be long to be long
    
    and the things I look the place
    
    i know I want you
    
    i got no be a show of your money so
    
    i don't know that she can she say hello
    
    when you say you say the end I'm so the dark
    
    why will you still the love and free
    
    in the morning more me mad
    
    i don't know that it's going to make you
    
    i can't be long long little girl
    
    if you know my name ha no need that hello
    
    it's so and on the time of time
    
    i can't be mine
    
    i think around for you
    
    you don't be long nothing you know that you do
    
    dear much a want to be your man before to me
    
    i'm going to lose you
    
    she does it ain't you have me too
    
    i want to be your man I love you
    
    i want to be your mother baby just can't be mine
    
    there's nothing you know my name
    
    but I know I nearly say hello
    
    the very and so mind you know
    
    that she do do you no should
    
    when I can't so boy
    
    i want you
    
    when you know I don't know my name
    
    i'm down I'm true to misey to mean a drink
    
    i think you want me
    
    and he said morning morning
    
    i'll be an the word of too much oh yeah
    
    that good and so
    
    all the lonely time is the sky
    
    the word she was a millow good
    
    the words and what it is you do
    
    all together now
    
    i think what you do
    
    i want to be your man me away
    
    the world of mind the word that she's happy just her help her
    
    we do it is so criend rings of love
    
    i want to see you that I can be hand
    
    i could love for your night and I'd sad you belong
    
    i love you yeah yeah yeah yeah
    
    and I will wear well the said
    
    the word is wondering
    
    i said she would do
    
    and I can see your money touce of mind
    
    we can be what I should gonna go ou
    
    she the coustion down the end
    
    i'm so home of soo much
    
    when you don't worry
    
    and I will see you love me do
    
    i feel you need you and me you're mine
    
    when I saw you can dead the could
    
    i think you know my name
    
    i can't come home to say
    
    all together now
    
    someday someone
    
    and I loved you
    
    and the word is a with on more to me
    
    it's my home
    
    don't be long with you on the number
    
    the calles in a lot to kiss you
    
    but you know what you know I heard before out of her back ago to me
    
    i'm gonna be the beginning
    
    i want to be your man I think you should
    
    you may see your mother should
    
    i want you
    
    you say all the windo that you see you
    
    you know I need you
    
    and I'm trying to make you
    
    and I will do it's only playents
    
    all together now
    
    he said so hard in the life of the start
    
    you know that you know my name
    
    what oh she would leave me in the light
    
    you'll be my mind
    
    for the baginning of the mystery ain't be silver right from the blistle help me
    
    you let me down you to let me know
    
    i'm gonna get to show that I'm round
    
    here got a little happened along
    
    i want you
    
    make you there mean me the word
    
    but the more will go
    
    oh yeah yeah yeah
    
    now time is here it won't be mine
    
    it's a loser
    
    the sun my walls here
    
    i feel you need to me
    
    if you can see that I want me
    
    ####################
    Temperature: 1.0
    ####################
    everyone plamis her eyes
    
    here can he can aperivad
    
    tell me away sad you can
    
    love me love new me the mack
    
    kissona without out someone
    
    just near right I know how man
    
    in you know my will
    
    well I'm stone the tartce play
    
    well you inter's in driverstile wind me
    
    'cause I'm love a woad a skilam
    
    the sun we're on a lover hole
    
    well you burn of you sunty face out
    
    i said it criend from shine
    
    we're looks to much the beginning
    
    she would daday new there with the night before and been the sun
    
    she love you toof
    
    christer writced time thationing
    
    i only hour them its and longer
    
    now no namr named everything you know
    
    come on the bird of good her
    
    but go on colourther knyings me tires came 
    
    because you know where yeah till I close are on I be
    
    i can't sergline
    
    oncs her life sun you leave me and mine
    
    ain't you know that I'm telling
    
    with a sooo on anyops
    
    take your head back 
    
    a look at the phone and knaw you silly stops I'm borwen
    
    where I fell the wrong to what you can no
    
    but you know I will the only 
    
    and now I feel a way
    
    she's got to die
    
    she world papper as she said
    
    love of all the little gight
    
    and it's true peop you know
    
    in the welt my boywen
    
    far had their is nowhere my window
    
    with you
    
    i know if I want hey wrong
    
    and you'll trupted to me
    
    here gone only feelon
    
    i said around down you know she won't be long
    
    where change I wright dose hear you
    
    honey my mine
    
    there going down
    
    but you know I nearly died
    
    they wetermed nor like you
    
    don't hidw the time without guy
    
    i want you you near yeah yeah yeah
    
    and then bant tonight
    
    oh when I saw you didn't know without you
    
    then you're too much what I see a
    
    hello looked you make 
    
    it's only in everything I'll ckild 
    
    can't you see now tire the apaparazine 
    
    in the love everyone frirty
    
    help you trive to me
    
    wonders do it apppear dayis has go words and hurther 
    
    and I found time the oughtle no come 
    
    so let you need is toptee another disappenebt never giving on up the show
    
    come on let't papperback man
    
    looking and I'll never make you
    
    remember she feel
    
    when I get you like tomorrow me
    
    beby cry beep me writer to a par
    
    all at the old on my mind of the sun
    
    i'm gonna get a fut home
    
    good make it all you need
    
    since's you're reding nowhere to make you and I love you
    
    so know I singing wo man't wind and fine
    
    all your neah you
    
    befper child she's be my one
    
    then I'm going take from my right me
    
    imside in the end
    
    it looking in her
    
    good us shoulder I'm in silver
    
    you you say you mather bast broke
    
    always fine me thinking of you
    
    oh yesterday to so hurmas
    
    a baby than the other
    
    see he take you much and todays more to play
    
    the his he walkad gone
    
    i don't let you know I won't all you need
    
    you'd gonna change goo gloou
    
    when I really his were
    
    it's herees born to be out cool
    
    come on uncuyfoother had a fretend
    
    come on the blits are love me me
    
    her came it's giving walks low
    
    celloo hil in I love you
    
    in a mil out halt look
    
    lace it's coming on the carme
    
    don't give my will be dance ach don't lonely
    
    oh oh i h ane have evened belong
    
    so see you told me that I'm senthiee
    
    it only times at armout tomorrown
    
    look at home
    
    it's all the hatchere
    
    but she knows prease whoa I want to be in 
    
    everybody I'm going to show out it's all only girl that boy
    
    come together pack boy
    
    server all together tone of mysteand
    
    so baby that my finger people are I'm long
    
    you burner the sun song you
    
    honey didn't see you
    
    and nothing I sad it come love morning
    
    maded that some love and I'm a roa
    
    when I feel you dyes a chance that you go
    
    sitting roarshing someone or more
    
    have another sudt her as good on bad good love
    
    home for the little child
    
    and you just finting so the walrus
    
    when I so I feel good more
    
    may's a wes blue
    
    yes it way what weagh I'm kissing and ip
    
    help me that is when you go
    
    man't you realize it's real
    
    we'd have a pain to give is in a man summer
    
    don't let me real
    
    i need " bitdaged darling
    
    i'd inthing yight time to make me I'll be there
    
    i need to me
    
    take you out your near sin'm on minders
    
    told me why you know my naw
    
    who she said sun here had a said
    
    but is the eggingen appen when you're insted and love girl that I said
    
    you better better better them turns
    
    martmaliss appartion
    
    don't worret home to
    
    did ky were baby with I belong
    
    c'mon gives you
    
    baby in and do
    
    when the love any loned's bill spillacper
    
    i want you
    
    oh oh ano you silver eyes
    
    there's nothing some
    
    i am a boy had goesboot bag
    
    ah ah ah oh
    
    love thate's no near like a love
    
    and lears where you dranks meettion
    
    i don't need now
    
    all I gotta do day
    
    waiting down to give every just
    
    there day with as the egg insterance"
    
    one the little girl by mine
    
    come on his love is what you see the oce
    
    in mistereasplicter bloked to bill
    
    just natten goodbye oncy love with me
    
    elach you've got to say
    
    couldn't have anyold I'm heard in really stops in her
    
    she does at mind I'm gone
    
    she one is madh a dranger before to hide and me
    
    she will go it be
    
    you told me un without of yestalnzaling
    
    i've amilan guy you'ted hand before
    
    cry brring look at the evening one your flirtantion I've bean
    
    to please me coming baby 
    
    look at home
    
    she's got nothing you can good
    
    where you have me seem armand
    
    paperback little child
    
    is who it show home about me in more
    
    i sad all weight
    
    oh my tump every don't care the man
    
    come on letting he's your bot our provy
    
    get back me love day sexy have ahearts in the ruin
    
    i know you want you know how man
    
    oh ah no no
    
    but you know my name hate's halmie
    
    now about it here such yeah like a with owhere
    
    that's agoing there without alone
    
    could me
    
    if you'll never do
    
    came it on simpore away
    
    hey break it won't you hear you mum
    
    if there's anyo good read
    
    oh my changing the whaces she dog deads 
    
    the newser I said hey'd kad to give or your say hurry with just bag little girl
    
    everybody sunshine
    
    once bring of little should
    
    you know you see her hands inving a millex
    
    love me earned alone
    
    now I've minemed seeting all it tight
    
    where you treat me see it be
    
    love and friend out
    
    if you look wonate everything you belong
    
    sed inside you it's riverhin'
    
    so is yeah can't see a remelatly hearts for easy 
    
    torking out
    
    my love without that before the touce blue
    
    yes you know I've missteen con't much 
    
    mn the many with go
    
    oh no doe what's a rumb me had bom
    
    let it be
    
    anyctead is a just again out you
    
    have that it kissoni-borncoom gonna come on
    
    and I'm hof love is love is you see run
    
    ah now I needed your birthday
    
    you know you no ooh man ooh moo
    
    that's shave wile gonely one
    
    back where you're be cromning
    
    remember where you'd be but little girl
    
    it tell me alone good a
    
    before me poot better make
    
    martha will be narmown
    
    come on his way a cride
    
    and she was someone
    
    if I think when I was alone bring 
    
    oh pleasons in the sky name
    
    come reople asteantil alrights
    
    i say seed you stand after the party
    
    made is procters to wrong
    
    "mouls befora
    
    you'll be gonna wanger hord inside
    
    it the egg on
    
    i can't tell my smore but the gues
    
    that's that evening now na leave
    
    send you didn't nead me in the ga
    
    she longer me ow
    
    back bydible after my proviess in the sgound of loww
    
    little better to you to my feet of more
    
    see the things the wild floe pain his out
    
    but but if baby
    
    well you did it a fa-la bla-la held do
    
    hold me along how her dead
    
    good just not tell you down
    
    write that is in freen't alway
    
    spease in the light
    
    i me tall you sit compoad
    
    someone the sun 
    
    do
    
    that you know homeone no now
    
    she becked me mad to me
    
    christmas is so my would ret
    
    you know you no
    
    i know you i had aloight to real
    
    three I'm down I'm roadiy
    
    boy you're gonna-moom
    
    say you know my name nah on
    
    it's so things to get good
    
    i can't heep you were do
    
    but it's all the word porgines
    
    man polyong in the stree
    
    perase not kind the bibtwas
    
    i dig no can kndeen' night you make me hold
    
    i'm just a show
    
    as I new she changed your dosaine
    
    we baby cry
    
    seing me more day's girl
    
    all around hearter leave me man
    
    say to heaven mean to the blinks a face
    
    good nudibng so thing but time
    
    oh you feeling the cold
    
    so the stag
    
    how dreamers lough to wrong
    
    here someone we lose it love
    
    all together nows
    
    you say
    
    was it is will down you
    
    teddy's rolly fadling I'm twice good must to heee a said doing
    
    please let chile sail love
    
    a the mun and with in 
    
    here magicat feel be a pain't bed
    
    i want you 
    
    it baby head
    
    did I want up 
    
    someday you don't need
    
    and a right come our your caccoo
    
    if please don't have some now
    
    i got a sain and happy pie
    
    you shout to much eon't real me so my
    
    i sady looks the cill like you
    
    cry
    
    you know my name
    
    if you lifetime plesed
    
    can't you go our in the man I want to be in my had
    
    to be I've sween would low to be
    
    he i'm an telling the one in my win
    
    i'm down I'm silver blings in my eyes
    
    it's always to gonna go
    
    it's been to with on sounly away
    
    ristucion fastio of you
    
    you'll be in the nunging must plans
    
    somelises homebod that prease don't badak
    
    we eave it is my window girl poart
    
    and it's making bast singing like it looking that playing closed
    
    it's true you to smile
    
    i'm just come oor to me
    
    i couldn't be night and I've been is blue
    
    "werepartand she was born
    
    when I wask when I get you one love
    
    everywhere away
    
    sa-caling words on scrooution
    
    they heavy needs you and I'm going to chack to be the hilleges
    
    i think you love me a comfoas away
    
    tell me nowwhe could never babe get nob letter
    
    ob look yight been here's nothing to holk this back girl
    
    all you need man somehel's maken in the dinkes of you
    
    eight till me that you know me
    
    feel up out on the party
    


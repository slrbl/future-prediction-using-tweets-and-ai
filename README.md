# Big events prediction with tweets and Ai

## Virtual environment

```
$ pip install pipenv
$ pipenv install --deploy --system --dev
$ pipenv shell
```

## Usage

```
$python crystall_ball.py -h
usage: crystall_ball.py [-h] -f TRAIN_FROM -u TRAIN_UNTIL [-d FUTURE_DEPTH]
                        [-s STEPS] -m MODEL [-l LANGUAGE]

optional arguments:
  -h, --help            show this help message and exit
  -f TRAIN_FROM, --train_from TRAIN_FROM
                        Training data from date in "yyyy/mm/dd" format
  -u TRAIN_UNTIL, --train_until TRAIN_UNTIL
                        Training data until date in "yyyy/mm/dd" format
  -d FUTURE_DEPTH, --future_depth FUTURE_DEPTH
                        Number of days to predict
  -s STEPS, --steps STEPS
                        Number of steps
  -m MODEL, --model MODEL
                        Choose LSTM or FBProphet
  -l LANGUAGE, --language LANGUAGE
                        Language. Available languages are ar_all, de_all,
                        en_all, es_all, fr_all, id_all, ko_all, pt_all, ru_all
```

## Example

```
$ python crystall_ball.py -f "2020/06/01" -u "2020/12/01" -d 3 -s 60 -l en_all -m "lstm"
```

## Details
https://www.linkedin.com/pulse/future-predictions-tweets-lstm-networks-2021-crystal-ball-daboubi/

set -x

if      [ ${1} = 'start' ]; then
    python 1.py



elif    [ ${1} = 'pull' ]; then
    #git clone -b develop ssh://git@git.sankuai.com/~shenzhe05/gdp.git
    git pull origin master
    git log

elif    [ ${1} = 'com' ]; then
    git add --all
    git commit -m ${2}
    git push origin HEAD:master

elif    [ ${1} = 'reset' ]; then
    git reset --hard HEAD~5


elif    [ ${1} = 't' ]; then
    tensorboard --logdir=/Users/shen/Desktop/log --port 8123 --host=127.0.0.1
fi



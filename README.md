# NeurIRS2019DroneChallengeRL
Microsoft AirSim NeurIRS2019 DroneChallenge implemented with RL and Computer Vision. Multi diverse , multi national group of people from Reddit and other Deep Learning communities 


How to use Google Cloud Enviroment

Connect with SSH  (a new terminal appears) 
-> after that click settings icon on right corner to a new Instance
(we need 2 terminals)

on first terminal to start simulation

bash run_docker_image.sh "" training    OR  bash run_docker_image.sh "" training headless . Both seems same at this point


-----

on second terminal to code run


cd /home/ugurkanates97/theCode/NeurIRS2019DroneChallengeRL/src/source
python metest.py (for train code)
python meplay.py(new code I added for playtesting) 

It saves models if they get better in training then current scores. You can also download .dat (model) files to normal local PC to see how it reacts. You can also see situtation from debug logs and you can save pictures

I will explain other things if needed?

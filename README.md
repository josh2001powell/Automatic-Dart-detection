# Automatic-Dart-detection
Code to work with a single camera for scoring a game of darts


Rough Scheme:

1  -  Calibrate dartboard with matrix transform. Will 'overlap' with idealised dartboard zones.

2  -  Live video feed capturing dart hits using differnce between frames.

3  -  Location of dart touching board is found using furthest point to one side. (Needs to be accurate for darts near boundaries)

4  -  Dart Location is transformed with same Matrix as calibration. Location --> Scoring zone

5  -  Save the scores of each dart and subtract from total. [If 0 reached with double --> GAME END] 

6  -  Once 3 darts detected (or user input) switches to next player.    REPEAT

   

Implementing threading of live video feed??

Thread 1
Waiting for PLAYER 1 to throw their darts
    - constantly comparing a frame from each second

Thread 2
Check 2 consecutive frames to see if there's been dart detected
    - Dart detected can probably be quanitified by some localised cluster of white (difference between frames), this should eliminate shadows or lighting changes
    - If dart has been detected then start another thread for Location
    - Once 3 darts have been detected (or keyboard input), switch to waiting for PLAYER 2

Thread 3
Takes the frame where dart is believed to be detected and finds its location and then scores
    - (choose) Draw contour around dart or fit ellipse?? Narrow end will be the dart head.
    - (choose) If camera placed at left side of board then furthest left point of dart should be the end of it.   hardware solution
    - Take the location of the head of the dart and use LOC2SCORE.py to assign score to that dart.

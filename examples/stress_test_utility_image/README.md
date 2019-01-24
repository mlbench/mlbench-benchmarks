Utility Test Image
------------------

A test image that utilizes each worker node with 2% CPU load, without doing any machine learning or anything else.

This is useful for testing mlbench with an image that doesn't have a huge footprint (So it can be ran with several nodes locally in DIND) 
while still producing CPU load to generate metrics.
